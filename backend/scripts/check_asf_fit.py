"""Reproduce the ASF-from-eTIV derivation used at inference, and measure its cost.

The pickle expects 24 features including ASF (Atlas Scaling Factor). The
hospital app never collects ASF — it is imaging-derived — so serving recovers it
from eTIV. This script produces the numbers quoted in
``app/services/prediction.py`` so they can be checked rather than trusted, and so
the R^2 figure is available for the report.

**2026-09-24 real-data update:** the training set itself is now real OASIS data
(real_dataset_setup.md), not a synthetic generator, so the original "synthetic
vs real" skew this script measured no longer exists as a *source* mismatch --
both sides are now OASIS-derived. What's still worth checking: does deriving ASF
from eTIV (rather than reading the OASIS pool's own true ASF) change any
prediction on the training set, now that the model is the real-data pickle.

Run from the repository root:

    python backend/scripts/check_asf_fit.py
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SET = REPO_ROOT / "data" / "multimodal_dementia_dataset.csv"
REAL = REPO_ROOT / "data" / "oasis2_mri_clinical.xlsx"
QSVM = REPO_ROOT / "results" / "qsvm_model.pkl"

CLINICAL = ["MMSE", "ASF", "EDUC", "SES"]
MRI = ["nWBV", "eTIV"]
SPEECH = [
    "pause_rate",
    "speech_rate",
    "pitch_mean",
    "jitter",
    "shimmer",
    *[f"mfcc_{i}" for i in range(1, 14)],
]
FEATURE_ORDER = CLINICAL + MRI + SPEECH
ASF_INDEX = FEATURE_ORDER.index("ASF")


def fit_constant(name: str, etiv, asf) -> float:
    """Least-squares fit of ASF ~= C / eTIV through the origin."""
    etiv = np.asarray(etiv, dtype=float)
    asf = np.asarray(asf, dtype=float)
    keep = np.isfinite(etiv) & np.isfinite(asf)
    etiv, asf = etiv[keep], asf[keep]

    inverse = 1.0 / etiv
    constant = float((inverse @ asf) / (inverse @ inverse))

    predicted = constant * inverse
    ss_res = float(((asf - predicted) ** 2).sum())
    ss_tot = float(((asf - asf.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot
    mae = float(np.abs(asf - predicted).mean())

    print(f"  {name:22} n={len(asf):4d}  C={constant:9.4f}  R^2={r_squared:+.6f}  MAE={mae:.5f}")
    return constant


def main() -> None:
    print("\n1. Fit ASF = C / eTIV")
    training = pd.read_csv(TRAINING_SET)
    fit_constant("training set (real, windowed)", training["eTIV"], training["ASF"])

    real = pd.read_excel(REAL)
    real_constant = fit_constant("real OASIS-2 (full)", real["eTIV"], real["ASF"])

    print(
        "\n  The training set is now real OASIS data too (its ASF column is copied\n"
        "  straight from oasis_pool.csv, not synthesised), so this fit should also be\n"
        "  ~exact -- OASIS derives eTIV from the atlas scaling factor, ASF = C/eTIV is\n"
        "  an identity on both the full OASIS-2 export and the matched training subset.\n"
        "  There is no more synthetic-vs-real skew from this source; ASF_ETIV_CONST\n"
        "  (1755.0, hardcoded in prediction.py) should be close to both C values above."
    )

    print("\n2. Does deriving ASF (vs. reading the OASIS pool's true ASF) change any prediction?")
    with QSVM.open("rb") as handle:
        pipeline = pickle.load(handle)

    features = training[FEATURE_ORDER].to_numpy(dtype=float)
    labels = training["Label"].to_numpy()
    baseline = pipeline.predict(features)

    derived = features.copy()
    derived[:, ASF_INDEX] = real_constant / derived[:, FEATURE_ORDER.index("eTIV")]
    derived_predictions = pipeline.predict(derived)

    constant_row = features.copy()
    constant_row[:, ASF_INDEX] = training["ASF"].mean()
    constant_predictions = pipeline.predict(constant_row)

    print(f"  true ASF (from pool)  accuracy {np.mean(baseline == labels):.4f}")
    print(
        f"  derived ASF (served)  accuracy {np.mean(derived_predictions == labels):.4f}"
        f"   flips {int((baseline != derived_predictions).sum())}/{len(labels)}"
    )
    print(
        f"  constant ASF          accuracy {np.mean(constant_predictions == labels):.4f}"
        f"   flips {int((baseline != constant_predictions).sum())}/{len(labels)}"
    )
    print(
        "\n  Flip count above is the served-vs-true ASF impact on this real training set --\n"
        "  see the printed numbers, not an assumed zero, for what actually happened.\n"
    )


if __name__ == "__main__":
    main()
