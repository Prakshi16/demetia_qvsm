"""Write results/model_info.json, the data behind the visit screens' model toggle.

For each served model (classical SVM, QSVM) it records:

* the fused 24-feature row's cross-validated metrics from
  ``results/metrics_full.json`` (5-fold GroupKFold, the capstone headline), and
* ``decision_scale``: the 95th percentile of |decision_function| over the
  training rows. The two models' decision values live on very different scales
  (an RBF-SVM margin vs a quantum-kernel margin), so the chart draws each
  patient's value against its own model's typical range. Without that, the
  toggle animation would mostly show a change of units.

Run from the repo root after any retrain:

    python backend/scripts/build_model_info.py
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
DATASET = REPO / "data" / "multimodal_dementia_dataset.csv"

MODELS = {
    "svm": {
        "name": "Classical SVM",
        "kernel": "RBF kernel",
        "canonical": True,
        "pickle": "svm_model.pkl",
        "metrics_key": "fused|svm",
    },
    "qsvm": {
        "name": "Quantum SVM",
        "kernel": "ZZFeatureMap fidelity kernel, 6 qubits (statevector simulation)",
        "canonical": False,
        "pickle": "qsvm_model.pkl",
        "metrics_key": "fused|qsvm",
    },
}


def main() -> int:
    metrics_full = json.loads((RESULTS / "metrics_full.json").read_text())
    runs = metrics_full["eval2_runs"]
    data = pd.read_csv(DATASET)
    X = data.iloc[:, 1:25].to_numpy(dtype=float)

    cards = []
    for key, spec in MODELS.items():
        run = runs.get(spec["metrics_key"])
        if run is None:
            raise SystemExit(f"{spec['metrics_key']} missing from metrics_full.json: {list(runs)}")
        with (RESULTS / spec["pickle"]).open("rb") as handle:
            model = pickle.load(handle)
        started = time.perf_counter()
        decision = np.abs(model.decision_function(X))
        per_row_ms = (time.perf_counter() - started) * 1000 / len(X)
        cards.append(
            {
                "key": key,
                "name": spec["name"],
                "kernel": spec["kernel"],
                "canonical": spec["canonical"],
                "metrics": {
                    name: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)}
                    for name, v in run["metrics"].items()
                },
                "decision_scale": round(float(np.percentile(decision, 95)), 4),
            }
        )
        print(f"{key}: decision_scale={cards[-1]['decision_scale']}  (~{per_row_ms:.1f} ms/row batched)")

    out = {
        "evaluation": f"{metrics_full['cv']}, fused 24-feature row, N={metrics_full['eval2_n']} real patients",
        "models": cards,
    }
    (RESULTS / "model_info.json").write_text(json.dumps(out, indent=2))
    print("wrote results/model_info.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
