"""Train and save the classical-SVM comparison model (results/svm_model.pkl).

Why this exists
---------------
Phase 1 pickled only the QSVM (`results/qsvm_model.pkl`). The app shows the
doctor TWO predictions side by side — "Quantum SVM" and "Classical SVM" — so the
Phase 1 headline finding (the quantum kernel shows no advantage over a classical
RBF SVM) is visible in the product itself rather than hidden. That needs a second
pickled model, which this script produces.

Design: the classical pipeline is IDENTICAL to the quantum one except for the
final estimator, so the two predictions are genuinely comparable and any
difference is attributable to the kernel, not to preprocessing:

    24 raw features -> ColumnTransformer(per-modality StandardScaler + PCA(2))
                    -> SVC(kernel='rbf')          [classical]
                    -> MinMaxScaler + QSVC        [quantum, from Phase 1]

The MinMaxScaler(0,1) step is a *quantum* requirement (raw PCA values alias past
2*pi inside the ZZFeatureMap and collapse the kernel); the classical RBF SVM does
not use it, matching the Phase 1 architecture exactly.

Trained on the real dataset (real_dataset_setup.md) — no SIGMA_FRAC noise. Real
data is not trivially separable the way the old synthetic set was (see the leak
probe in scripts/verify_real_dataset.py), so a clean fit is already the honest
number; noise injection is not needed and must never be applied at inference.

Cross-validation is GroupKFold(5) on the composite `speaker_id|oasis_subject_id`
group (read from `data/multimodal_real_provenance.csv`), matching
`multimodal_qsvm.ipynb` / `metrics_full.ipynb` — StratifiedKFold would let one
speaker's windows span train and test.

Run from the repository root:

    python backend/scripts/train_classical_svm.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# scripts/ -> backend/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "multimodal_dementia_dataset.csv"
PROVENANCE = REPO_ROOT / "data" / "multimodal_real_provenance.csv"
OUTPUT = REPO_ROOT / "results" / "svm_model.pkl"

# The exact Phase 1 feature order (real-data, 27->24 migration; CDR dropped, MRI
# reduced to the 2 real tabular columns). Column indices below depend on it.
CLINICAL_FEATURES = ["MMSE", "ASF", "EDUC", "SES"]
MRI_FEATURES = ["nWBV", "eTIV"]
SPEECH_FEATURES = [
    "pause_rate",
    "speech_rate",
    "pitch_mean",
    "jitter",
    "shimmer",
    *[f"mfcc_{i}" for i in range(1, 14)],
]
FEATURE_ORDER = CLINICAL_FEATURES + MRI_FEATURES + SPEECH_FEATURES

CLINICAL_IDX = list(range(0, 4))
MRI_IDX = list(range(4, 6))
SPEECH_IDX = list(range(6, 24))


def build_pipeline() -> Pipeline:
    """The classical twin of the Phase 1 quantum pipeline."""
    modal = ColumnTransformer(
        transformers=[
            (
                "clinical",
                Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=2))]),
                CLINICAL_IDX,
            ),
            (
                "mri",
                Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=2))]),
                MRI_IDX,
            ),
            (
                "speech",
                Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=2))]),
                SPEECH_IDX,
            ),
        ]
    )
    return Pipeline([("modal", modal), ("clf", SVC(kernel="rbf", C=1.0, gamma="scale"))])


def main() -> None:
    frame = pd.read_csv(DATASET)
    provenance = pd.read_csv(PROVENANCE)[["Subject_ID", "group_id"]]
    frame = frame.merge(provenance, on="Subject_ID", how="left")
    assert frame["group_id"].isna().sum() == 0, "every row must carry a composite group_id"

    features = frame[FEATURE_ORDER].to_numpy(dtype=float)
    labels = frame["Label"].to_numpy()
    groups = frame["group_id"].to_numpy()

    assert features.shape[1] == 24, f"expected 24 features, got {features.shape[1]}"

    pipeline = build_pipeline()

    # GroupKFold(5) on the composite speaker|oasis_subject group -- matches
    # multimodal_qsvm.ipynb / metrics_full.ipynb exactly. This IS the honest
    # headline number now (real data, no noise needed).
    scores = cross_val_score(
        pipeline,
        features,
        labels,
        cv=GroupKFold(n_splits=5),
        groups=groups,
    )
    print(f"5-fold GroupKFold CV accuracy : {scores.mean():.4f} +/- {scores.std():.4f}")

    pipeline.fit(features, labels)
    print(f"train accuracy           : {(pipeline.predict(features) == labels).mean():.4f}")
    print(f"support vectors          : {pipeline.named_steps['clf'].n_support_}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as handle:
        pickle.dump(pipeline, handle)

    print(f"saved                    : {OUTPUT.relative_to(REPO_ROOT)}")

    # Round-trip check: the served model must load and predict identically.
    with OUTPUT.open("rb") as handle:
        reloaded = pickle.load(handle)
    assert np.array_equal(reloaded.predict(features), pipeline.predict(features))
    print("round-trip verified      : OK")


if __name__ == "__main__":
    main()
