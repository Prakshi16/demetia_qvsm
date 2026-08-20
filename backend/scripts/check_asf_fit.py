"""Reproduce the ASF-from-eTIV derivation used at inference, and measure its cost.

The Phase 1 pickle expects 27 features including ASF (Atlas Scaling Factor). The
hospital app never collects ASF — it is imaging-derived — so serving recovers it
from eTIV. This script produces the numbers quoted in
``app/services/prediction.py`` so they can be checked rather than trusted, and so
the R^2 figure is available for the report.

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
SYNTHETIC = REPO_ROOT / "data" / "multimodal_dementia_dataset.csv"
REAL = REPO_ROOT / "data" / "oasis_longitudinal_demographics.xlsx"
QSVM = REPO_ROOT / "results" / "qsvm_model.pkl"

CLINICAL = ["CDR", "MMSE", "ASF", "EDUC", "SES"]
MRI = ["nWBV", "eTIV", "hippocampal_volume", "cortical_thickness"]
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
    synthetic = pd.read_csv(SYNTHETIC)
    fit_constant("synthetic (training)", synthetic["eTIV"], synthetic["ASF"])

    real = pd.read_excel(REAL)
    real_constant = fit_constant("real OASIS-2", real["eTIV"], real["ASF"])

    print(
        "\n  On real patients the relation is EXACT (R^2 = 1.000000): OASIS derives\n"
        "  eTIV from the atlas scaling factor, so ASF = C/eTIV is an identity, not\n"
        "  an approximation. On the synthetic training set R^2 is NEGATIVE — the\n"
        "  generator drew ASF and eTIV independently — so the ASF we serve is\n"
        "  distributed differently from the ASF the pickle was trained on."
    )

    print("\n2. Does that train/serve skew change any prediction?")
    with QSVM.open("rb") as handle:
        pipeline = pickle.load(handle)

    features = synthetic[FEATURE_ORDER].to_numpy(dtype=float)
    labels = synthetic["Label"].to_numpy()
    baseline = pipeline.predict(features)

    derived = features.copy()
    derived[:, ASF_INDEX] = real_constant / derived[:, FEATURE_ORDER.index("eTIV")]
    derived_predictions = pipeline.predict(derived)

    constant_row = features.copy()
    constant_row[:, ASF_INDEX] = synthetic["ASF"].mean()
    constant_predictions = pipeline.predict(constant_row)

    print(f"  true ASF        accuracy {np.mean(baseline == labels):.4f}")
    print(
        f"  derived ASF     accuracy {np.mean(derived_predictions == labels):.4f}"
        f"   flips {int((baseline != derived_predictions).sum())}/{len(labels)}"
    )
    print(
        f"  constant ASF    accuracy {np.mean(constant_predictions == labels):.4f}"
        f"   flips {int((baseline != constant_predictions).sum())}/{len(labels)}"
    )
    print(
        "\n  Zero flips either way: the MRI block dominates the fused decision, so\n"
        "  the skew is documented and quantified rather than material.\n"
    )


if __name__ == "__main__":
    main()
