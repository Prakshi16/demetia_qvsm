"""Calibrate the live MRI extractor's nWBV/eTIV output against real OASIS-1 ground truth.

Why this exists (2026-09-25): testing the app with real OASIS scans found that
backend/app/services/mri_features.py's raw heuristic (Otsu-threshold intracranial mask,
20th-percentile-of-mask "brain" cutoff) badly compresses real atrophy differences -- three real
subjects with true nWBV 0.83/0.76/0.67 all extracted to ~0.85-0.88 through the raw heuristic,
flipping predictions. See PROJECT_CONTEXT.md for the full diagnosis.

Approach: we have real ground truth on disk. 39 OASIS-1 subjects have both a processed,
skull-stripped scan (data/oasis1_MRI_scans/OAS1_*_MR1/PROCESSED/MPRAGE/T88_111/*_masked_gfc.img)
and a true nWBV/eTIV in the raw OASIS-1 xlsx (nWBV/eTIV don't depend on CDR being non-null, so
this uses every on-disk subject, not just the ~25 with a defined dementia label). For each scan we
compute the same feature vector the live extractor now shares
(app.services.mri_features.compute_calibration_features -- the existing heuristic outputs plus a
handful of intensity-percentile mask-volume ratios) and fit two small Ridge regressions (one for
nWBV, one for eTIV) mapping that feature vector to OASIS's true values.

N=39 is small, so: Ridge (regularized, not a flexible model that would overfit 39 points), alpha
chosen by leave-one-out CV (the honest, standard choice at this sample size), and the LOOCV
MAE/R^2 reported here are the real, disclosed accuracy ceiling of this calibration -- not a claim
of clinical-grade accuracy. Still a proxy, not real tissue segmentation.

Run from the repository root:

    python scripts/calibrate_mri_extractor.py

Writes results/mri_extractor_calibration.pkl (loaded by mri_features.py's _load_calibration()).
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.mri_features import (  # noqa: E402
    CALIBRATION_FEATURE_NAMES,
    calibration_feature_vector,
    extract_calibration_features_from_path,
)

SCAN_ROOT = REPO_ROOT / "data" / "oasis1_MRI_scans"
CLINICAL_XLSX = REPO_ROOT / "data" / "oasis1_clinical data.xlsx"
OUTPUT = REPO_ROOT / "results" / "mri_extractor_calibration.pkl"

ALPHAS = np.logspace(-2, 3, 30)


def _find_masked_scan(subject_dir: Path) -> Path | None:
    t88_dir = subject_dir / "PROCESSED" / "MPRAGE" / "T88_111"
    if not t88_dir.exists():
        return None
    matches = sorted(t88_dir.glob("*_masked_gfc.img"))
    return matches[0] if matches else None


def _build_calibration_table() -> pd.DataFrame:
    clinical = pd.read_excel(CLINICAL_XLSX)
    clinical["_subject_id"] = clinical["ID"].astype(str).str.replace(r"_MR\d+$", "", regex=True)
    clinical = clinical.drop_duplicates("_subject_id", keep="first").set_index("_subject_id")

    rows = []
    for subject_dir in sorted(SCAN_ROOT.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = re.sub(r"_MR\d+$", "", subject_dir.name)
        scan_path = _find_masked_scan(subject_dir)
        if scan_path is None:
            print(f"  skip {subject_id}: no processed/masked scan on disk")
            continue
        if subject_id not in clinical.index:
            print(f"  skip {subject_id}: not found in {CLINICAL_XLSX.name}")
            continue

        true_nwbv = clinical.loc[subject_id, "nWBV"]
        true_etiv = clinical.loc[subject_id, "eTIV"]
        if pd.isna(true_nwbv) or pd.isna(true_etiv):
            print(f"  skip {subject_id}: missing true nWBV/eTIV")
            continue

        features = extract_calibration_features_from_path(scan_path)
        rows.append(
            {
                "subject_id": subject_id,
                "true_nwbv": float(true_nwbv),
                "true_etiv": float(true_etiv),
                **features,
            }
        )

    return pd.DataFrame(rows)


def _loocv(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, float, np.ndarray]:
    """Manual leave-one-out CV for one alpha. Small N (39) makes this cheap and fully transparent
    (vs. relying on RidgeCV's internal GCV machinery)."""
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = Ridge(alpha=alpha)
        model.fit(X[mask], y[mask])
        preds[i] = model.predict(X[i : i + 1])[0]
    mae = float(np.abs(preds - y).mean())
    ss_res = float(((preds - y) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, r2, preds


def _fit_target(X: np.ndarray, y: np.ndarray, name: str) -> tuple[Ridge, dict]:
    scored = [(alpha, _loocv(X, y, alpha)) for alpha in ALPHAS]
    best_alpha, (best_mae, best_r2, _) = min(scored, key=lambda t: t[1][0])
    print(f"  {name:6} best alpha={best_alpha:8.3f}  LOOCV MAE={best_mae:.4f}  LOOCV R^2={best_r2:+.3f}")
    final_model = Ridge(alpha=best_alpha).fit(X, y)
    stats = {"alpha": float(best_alpha), "loocv_mae": best_mae, "loocv_r2": best_r2, "n": len(y)}
    return final_model, stats


def main() -> int:
    print(f"1. Building the calibration table from {SCAN_ROOT.relative_to(REPO_ROOT)}")
    df = _build_calibration_table()
    print(f"\nCalibration set: {len(df)} OASIS-1 subjects with a processed scan + true nWBV/eTIV")
    if len(df) < 10:
        print("Too few calibration subjects found -- check data/oasis1_MRI_scans/.")
        return 1

    print("\n2. Baseline: the raw (uncalibrated) heuristic's own error on this set")
    old_nwbv_mae = float((df["nwbv_heuristic"] - df["true_nwbv"]).abs().mean())
    old_etiv_mae = float((df["icv_ml"] - df["true_etiv"]).abs().mean())
    print(f"  nwbv (raw heuristic)  MAE={old_nwbv_mae:.4f}")
    print(f"  eTIV (raw ICV heuristic)  MAE={old_etiv_mae:.2f}")

    print("\n3. Fitting calibrated Ridge regressions (LOOCV alpha selection)")
    X = df[list(CALIBRATION_FEATURE_NAMES)].to_numpy(dtype=float)
    nwbv_model, nwbv_stats = _fit_target(X, df["true_nwbv"].to_numpy(dtype=float), "nWBV")
    etiv_model, etiv_stats = _fit_target(X, df["true_etiv"].to_numpy(dtype=float), "eTIV")

    print(
        f"\nImprovement: nWBV MAE {old_nwbv_mae:.4f} -> {nwbv_stats['loocv_mae']:.4f}, "
        f"eTIV MAE {old_etiv_mae:.2f} -> {etiv_stats['loocv_mae']:.2f}"
    )

    calibration = {
        "feature_names": CALIBRATION_FEATURE_NAMES,
        "nwbv_model": nwbv_model,
        "etiv_model": etiv_model,
        "nwbv_stats": nwbv_stats,
        "etiv_stats": etiv_stats,
        "n_calibration_subjects": len(df),
        "calibration_subject_ids": sorted(df["subject_id"].tolist()),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as handle:
        pickle.dump(calibration, handle)
    print(f"\nSaved {OUTPUT.relative_to(REPO_ROOT)}")

    # Round-trip check, mirroring train_classical_svm.py's convention.
    with OUTPUT.open("rb") as handle:
        reloaded = pickle.load(handle)
    row = calibration_feature_vector(df.iloc[0][list(CALIBRATION_FEATURE_NAMES)].to_dict())
    assert reloaded["nwbv_model"].predict([row])[0] == nwbv_model.predict([row])[0]
    print("Round-trip verified: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
