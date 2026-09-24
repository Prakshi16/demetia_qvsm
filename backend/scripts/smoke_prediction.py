"""Smoke-test the model-serving seam without a database or an HTTP request.

Exercises exactly what an upload endpoint triggers — feature assembly, ASF
derivation, both pipelines, and the visit state transition — using two real
patients from the real assembled dataset (real_dataset_setup.md), embedded as
literals so this runs for anyone (``data/`` is gitignored and teammates don't
have it). Both rows were confirmed against the 2026-09-24 retrained pickles
(``results/qsvm_model.pkl`` / ``results/svm_model.pkl``) to predict their true
label under the *served* feature path (ASF derived from eTIV, not read from the
CSV — see ``prediction.py``'s ASF note) before being hardcoded here.

Run inside the backend container, where the pickles are mounted at /model:

    docker compose run --rm backend python scripts/smoke_prediction.py

Expected: both patients predicted correctly, status advances to
``pending_review``, and each prediction takes roughly half a second.
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from pathlib import Path

# Run from anywhere: put backend/ (the parent of scripts/) on the import path so
# `app` resolves without needing PYTHONPATH set.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import Visit  # noqa: E402
from app.services import prediction as P  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

# Two real rows from data/multimodal_dementia_dataset.csv (real, windowed OASIS x speech
# match). `cdr` is display-only (not a model input any more -- see prediction.py's CDR
# note) and is left unset here; the model never reads it.
PATIENTS = [
    {
        "name": "oasis1_OAS1_0366_w0 -- Label=0 (Nondemented)",
        "truth": "Nondemented",
        "clinical": {"mmse": 29.0, "edu": 18.0, "ses": 2.0},
        "mri": [0.813, 1549.0],
        "speech": [
            0.2004, 5.7, 164.53, 0.0322, 0.1819,
            -384.251, 98.5568, 20.7769, 19.3264, -10.1216, 1.6509, -9.008,
            1.4172, -5.835, 0.7632, 2.1167, 4.4212, 8.7332,
        ],
    },
    {
        "name": "oasis1_OAS1_0041_w0 -- Label=1 (Demented)",
        "truth": "Demented",
        "clinical": {"mmse": 28.0, "edu": 12.0, "ses": 3.1577},
        "mri": [0.758, 1350.0],
        "speech": [
            0.2004, 4.2, 197.8473, 0.0669, 0.1761,
            -237.7371, 104.5047, -4.1745, 17.4252, -10.0145, -2.6665, -8.6135,
            -8.8318, -5.3759, -6.6223, -4.2719, -3.0018, 0.4886,
        ],
    },
]


def build_visit(spec: dict) -> Visit:
    """A screening visit in the exact state an upload endpoint leaves behind."""
    visit = Visit()
    visit.id = uuid.uuid4()
    visit.visit_type = "screening"
    visit.status = "awaiting_uploads"
    visit.mri_status = "done"
    visit.speech_status = "done"
    visit.mri_feature_vector = spec["mri"]
    visit.speech_feature_vector = spec["speech"]
    visit.model_prediction = None
    for field, value in spec["clinical"].items():
        setattr(visit, field, value)
    return visit


def main() -> int:
    failures: list[str] = []

    print(f"\nmodel dir : {P._model_dir()}")
    print(f"USE_REAL_MODEL : {P.USE_REAL_MODEL}")

    started = time.perf_counter()
    if not P.warm_models():
        print("\nFAIL: models did not load — check the /model mount and the pickles.")
        return 1
    print(f"warm_models    : ok in {time.perf_counter() - started:.2f}s\n")

    for spec in PATIENTS:
        visit = build_visit(spec)

        row = P.build_feature_row(visit)
        if len(row) != 24:
            failures.append(f"{spec['name']}: feature row was {len(row)}, expected 24")

        started = time.perf_counter()
        fired = P.check_and_run_prediction(None, visit)
        elapsed = time.perf_counter() - started

        print(f"{spec['name']}")
        print(f"  ASF derived from eTIV : {row[1]:.4f}  (eTIV {row[5]:.1f})")
        print(f"  fired                 : {fired}   status -> {visit.status}")
        print(f"  Quantum SVM           : {visit.model_prediction}  margin {visit.model_confidence:.4f}")
        print(f"  Classical SVM         : {visit.svm_prediction}  margin {visit.svm_confidence:.4f}")
        print(f"  truth                 : {spec['truth']}")
        print(f"  latency               : {elapsed * 1000:.0f} ms\n")

        if not fired:
            failures.append(f"{spec['name']}: prediction did not fire")
        if visit.status != "pending_review":
            failures.append(f"{spec['name']}: status is {visit.status}, expected pending_review")
        if visit.model_prediction != spec["truth"]:
            failures.append(
                f"{spec['name']}: QSVM said {visit.model_prediction}, expected {spec['truth']}"
            )

    # Idempotence: a second call must not re-run or clobber an existing prediction.
    visit = build_visit(PATIENTS[0])
    P.check_and_run_prediction(None, visit)
    first = visit.model_prediction
    if P.check_and_run_prediction(None, visit) is not False:
        failures.append("re-running on an already-predicted visit should return False")
    if visit.model_prediction != first:
        failures.append("re-running clobbered an existing prediction")

    # A follow-up visit must never trigger the model.
    follow_up = build_visit(PATIENTS[0])
    follow_up.visit_type = "follow_up"
    if P.check_and_run_prediction(None, follow_up) is not False:
        failures.append("a follow-up visit must not run the model")

    # An incomplete visit must never trigger the model.
    incomplete = build_visit(PATIENTS[0])
    incomplete.mri_status = "idle"
    if P.check_and_run_prediction(None, incomplete) is not False:
        failures.append("a visit missing MRI must not run the model")

    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("All checks passed: predictions correct, guards hold, state advances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
