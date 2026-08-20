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

    27 raw features -> ColumnTransformer(per-modality StandardScaler + PCA(2))
                    -> SVC(kernel='rbf')          [classical]
                    -> MinMaxScaler + QSVC        [quantum, from Phase 1]

The MinMaxScaler(0,1) step is a *quantum* requirement (raw PCA values alias past
2*pi inside the ZZFeatureMap and collapse the kernel); the classical RBF SVM does
not use it, matching the Phase 1 architecture exactly.

Trained on CLEAN synthetic data — no SIGMA_FRAC noise. The Phase 1 noise
injection existed to produce an honest cross-validated *evaluation* number on a
trivially separable dataset; it is a property of the evaluation, not of the
served model. The quantum pickle was likewise fit on clean data (it scores 1.000
on the full synthetic set). Never apply inference-time noise.

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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# scripts/ -> backend/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "data" / "multimodal_dementia_dataset.csv"
OUTPUT = REPO_ROOT / "results" / "svm_model.pkl"

# The exact Phase 1 feature order. Column indices below depend on it.
CLINICAL_FEATURES = ["CDR", "MMSE", "ASF", "EDUC", "SES"]
MRI_FEATURES = ["nWBV", "eTIV", "hippocampal_volume", "cortical_thickness"]
SPEECH_FEATURES = [
    "pause_rate",
    "speech_rate",
    "pitch_mean",
    "jitter",
    "shimmer",
    *[f"mfcc_{i}" for i in range(1, 14)],
]
FEATURE_ORDER = CLINICAL_FEATURES + MRI_FEATURES + SPEECH_FEATURES

CLINICAL_IDX = list(range(0, 5))
MRI_IDX = list(range(5, 9))
SPEECH_IDX = list(range(9, 27))


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
    features = frame[FEATURE_ORDER].to_numpy(dtype=float)
    labels = frame["Label"].to_numpy()

    assert features.shape[1] == 27, f"expected 27 features, got {features.shape[1]}"

    pipeline = build_pipeline()

    # Reported for the record only — this is clean, trivially separable synthetic
    # data, so it is NOT the honest headline number. The defensible evaluation is
    # the noisy 5-fold CV in results/multimodal_results.json.
    scores = cross_val_score(
        pipeline,
        features,
        labels,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
    )
    print(f"clean 5-fold CV accuracy : {scores.mean():.4f} +/- {scores.std():.4f}")

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
