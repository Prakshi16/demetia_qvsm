"""Model-serving seam for the fused prediction (classical SVM + QSVM research view).

**2026-11-04: QSVM back as a research view.** For the capstone's academic
comparison, the visit screens now have a model toggle that shows the QSVM's
prediction for the same patient next to the classical SVM's. The QSVM is computed
alongside the SVM into its own columns (``qsvm_prediction``/``qsvm_confidence``,
migration 006). **The classical SVM stays canonical:** ``model_prediction`` is
still its result and ``agreement_flag`` still compares the doctor against it.
A QSVM failure only leaves the research columns empty; it never blocks the
canonical prediction or the clinician's queue. Cost: about 0.5 s of statevector
kernel evaluation per prediction, inside the second upload request.

``check_and_run_prediction`` is the exact function Bishal's MRI upload and
Sheetal's speech upload endpoints call at the very end of their handlers, once
they've stored their feature vector and set their modality status to ``done``.
When BOTH modalities are done it runs the fused pipeline and moves the visit into
the clinician's queue (``status='pending_review'``).

**2026-09-25 — QSVM removed from serving (decision, not an oversight).** The
app used to run both the QSVM and a classical SVM and show them side by side,
with the QSVM as canonical (``agreement_flag`` compared the doctor against it).
On the real-data retrain (``real_dataset_setup.md`` / ``results/metrics_full.json``,
5-fold ``GroupKFold``, fused), the classical SVM beat the QSVM on every metric
but specificity — accuracy 81.1% vs 78.1%, F1 83.4% vs 79.9%, and ROC-AUC 90.8%
vs 84.5% (the biggest gap, meaning its confidence ranking is meaningfully more
trustworthy, not just its hard label) — and sensitivity 85.9% vs 79.4%, which
matters more for a screening tool since missing a real case is the costlier
error. So the app now loads and runs **only** the classical SVM, and its result
fills ``model_prediction``/``model_confidence`` (the field ``agreement_flag``
compares the doctor against). ``svm_prediction``/``svm_confidence`` are no
longer populated for new predictions — the historical QSVM comparison columns
stay in the schema for old visits, they just aren't written to any more.
**The QSVM is not deleted from the project** — ``multimodal_qsvm.ipynb`` and
``metrics_full.ipynb`` still train and evaluate it for the capstone's honest
"does the quantum kernel help?" comparison; this file is the one place that
stopped calling it in the live product.

=============================================================================
SERVING NOTES (decisions resolved in the model-serving pass)

* **Latency.** Removing the QSVM roughly halves per-prediction latency: the
  QSVM's ``FidelityStatevectorKernel`` evaluation was measured at **~445 ms**
  locally (146 support vectors) — the slow half of the old two-model call.
  Prediction stays SYNCHRONOUS either way — comfortably inside a normal
  request. The pickle is loaded once at startup, not per request.

* **Confidence is a MARGIN, not a probability.** The classical ``SVC`` is fit
  with ``probability=False`` and exposes no ``predict_proba``, so this reports
  ``abs(decision_function(...))`` — signed distance from the separating
  hyperplane. It is NOT calibrated and must never be shown to a clinician as a
  percentage likelihood. One ``decision_function`` call also yields the label
  from its sign.

* **ASF is derived from eTIV.** The pickle wants 24 features including ``ASF``
  (Atlas Scaling Factor), which the app never collects. See ``ASF_ETIV_CONST``
  below for the fit and its measured impact.

* **CDR is not a model input (2026-09-24, real-data migration).** OASIS's dementia
  label is defined *from* CDR, so feeding CDR to the model would re-inject the
  label as a feature. The app still collects/stores ``visit.cdr`` for clinical
  display (see ``patients.py``'s MMSE/CDR trend), it just isn't part of
  ``build_feature_row`` any more. Clinical block is now ``[MMSE, ASF, EDUC, SES]``
  (4, not 5) and the MRI block is ``[nWBV, eTIV]`` (2, not 4 —
  ``hippocampal_volume``/``cortical_thickness`` were synthetic-only estimates with
  no OASIS tabular counterpart). Total row length: 24.

* **No inference-time noise.** Phase 1's ``SIGMA_FRAC`` noise injection was a
  property of the *evaluation* (it produced an honest CV number on trivially
  separable synthetic data). Applying it at inference would corrupt a real
  patient's prediction. It is deliberately absent here.
=============================================================================
"""
from __future__ import annotations

import logging
import pickle
import time
from functools import lru_cache
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Visit

logger = logging.getLogger(__name__)

# Flip to False to fall back to the deterministic stub (useful if a teammate is
# working without the pickles present).
USE_REAL_MODEL = True

# The classical SVM is the only model the app serves (see the 2026-09-25 note
# above). results/qsvm_model.pkl still exists for the research notebooks but is
# never loaded here.
SVM_PICKLE = "svm_model.pkl"
# Research view only (2026-11-04) — see the note at the top of this file.
QSVM_PICKLE = "qsvm_model.pkl"

# --------------------------------------------------------------------------
# ASF derivation
# --------------------------------------------------------------------------
# The Phase 1 pipeline expects clinical features [CDR, MMSE, ASF, EDUC, SES],
# but the app's clinical form collects only MMSE/CDR/EDUC/SES — ASF is an
# imaging-derived quantity, so we recover it from eTIV (which the MRI extractor
# does produce).
#
# Fitted as ASF ~= C / eTIV by least squares through the origin:
#   * real OASIS-2 (n=373): C = 1755.0, R^2 = 1.000000, MAE = 0.00000
#     -> on real patients this is not an approximation at all, it is an exact
#        identity: OASIS derives eTIV FROM the atlas scaling factor.
#   * The training set is now real OASIS data too (see real_dataset_setup.md), so
#     this constant is no longer a train/serve skew the way it was against the
#     old synthetic set — it is the same identity on both sides.
#
# Reproduce: backend/scripts/check_asf_fit.py
ASF_ETIV_CONST = 1755.0

# Fallbacks for nullable clinical columns (medians of the real training set,
# data/multimodal_dementia_dataset.csv, recomputed 2026-09-24 after the real-data
# retrain — see real_dataset_setup.md Appendix C). `edu`/`ses` are screening-only
# and nullable; a screening visit should always carry them, but the pipeline
# cannot accept NaN, so we impute rather than 500. CDR has no fallback here: it
# is not a model input (see the CDR note above).
EDUC_FALLBACK = 14.0
SES_FALLBACK = 2.31
MMSE_FALLBACK = 28.0

# Maps the pickle's integer classes onto the visits.model_prediction enum.
CLASS_LABELS = {0: "Nondemented", 1: "Demented"}


def _model_dir() -> Path:
    """Where the pickle lives.

    docker-compose mounts ./results at settings.MODEL_DIR (/model). Running
    uvicorn directly outside Docker, that path doesn't exist, so fall back to
    <repo>/results — services/ -> app/ -> backend/ -> repo root.
    """
    configured = Path(settings.MODEL_DIR)
    if (configured / SVM_PICKLE).exists():
        return configured
    return Path(__file__).resolve().parents[3] / "results"


@lru_cache(maxsize=1)
def _load_models() -> object:
    """Load the classical SVM pipeline once. Cached — never re-read per request.

    Required: this is the only model the app serves. Regenerate it with
    `python backend/scripts/train_classical_svm.py`.
    """
    directory = _model_dir()

    svm_path = directory / SVM_PICKLE
    with svm_path.open("rb") as handle:
        svm = pickle.load(handle)
    logger.info("Loaded classical SVM pipeline from %s", svm_path)

    return svm


@lru_cache(maxsize=1)
def _load_qsvm() -> object | None:
    """Load the QSVM pickle once, or None if it's missing or won't unpickle.

    Optional by design: it powers the research toggle only.
    """
    path = _model_dir() / QSVM_PICKLE
    if not path.exists():
        logger.warning("QSVM pickle missing at %s; the research view will be empty", path)
        return None
    try:
        with path.open("rb") as handle:
            qsvm = pickle.load(handle)
    except Exception:  # noqa: BLE001 - optional model, never fatal
        logger.exception("Could not load the QSVM pickle; the research view will be empty")
        return None
    logger.info("Loaded QSVM pipeline from %s", path)
    return qsvm


def qsvm_available() -> bool:
    return _load_qsvm() is not None


def warm_models() -> bool:
    """Preload the pickle at startup so the first upload isn't slow.

    Returns True if it loaded. Never raises: a model failure must not stop
    the API from booting — uploads still work, the visit just stays in
    `awaiting_uploads` instead of advancing, and /health reports the problem.
    """
    if not USE_REAL_MODEL:
        return False
    try:
        _load_models()
        _load_qsvm()  # optional; logs and returns None on failure
        return True
    except Exception:  # noqa: BLE001 - startup must survive any model failure
        logger.exception("Model preload failed; predictions will fall back to the stub")
        return False


def _first(value: float | None, fallback: float) -> float:
    return fallback if value is None else float(value)


def build_feature_row(visit: Visit) -> list[float]:
    """Assemble the 24-feature row in the exact real-data column order.

    [0:4]   clinical  MMSE, ASF, EDUC, SES   (CDR dropped -- see the CDR note above)
    [4:6]   mri       nWBV, eTIV
    [6:24]  speech    pause_rate, speech_rate, pitch_mean, jitter, shimmer,
                      mfcc_1 .. mfcc_13

    Kept public so it can be unit-tested without loading a pickle.
    """
    mri = [float(value) for value in (visit.mri_feature_vector or [])]
    speech = [float(value) for value in (visit.speech_feature_vector or [])]

    if len(mri) != 2:
        raise ValueError(f"MRI feature vector must have 2 values, got {len(mri)}")
    if len(speech) != 18:
        raise ValueError(f"Speech feature vector must have 18 values, got {len(speech)}")

    etiv = mri[1]
    if etiv <= 0:
        raise ValueError("eTIV must be positive to derive ASF")
    asf = ASF_ETIV_CONST / etiv

    clinical = [
        _first(visit.mmse, MMSE_FALLBACK),
        asf,
        _first(visit.edu, EDUC_FALLBACK),
        _first(visit.ses, SES_FALLBACK),
    ]

    row = clinical + mri + speech
    if len(row) != 24:
        raise ValueError(f"Expected a 24-feature row, built {len(row)}")
    return row


def _predict_with(pipeline: object, row: list[float]) -> tuple[str, float]:
    """One decision_function call -> (label, margin).

    Binary classifier, classes_ = [0, 1]: a positive decision value means class 1
    ("Demented"). Deriving the label from the sign avoids a second call that
    predict() would cost.
    """
    margin = float(pipeline.decision_function([row])[0])
    label = CLASS_LABELS[1] if margin > 0 else CLASS_LABELS[0]
    return label, abs(margin)


def clear_stale_prediction(visit: Visit) -> None:
    """Clear a previously-computed prediction so a corrected re-upload recomputes it.

    Called by mri_upload.py / speech.py, right before check_and_run_prediction, when a
    receptionist re-uploads MRI or speech after a prediction already fired (e.g. the wrong
    file was picked the first time). Without this, check_and_run_prediction's own guard
    ("already predicted, don't re-run") would silently leave the stale prediction in place
    even though the underlying evidence just changed. The router is responsible for refusing
    this once a diagnosis has been saved (visit.status == "reviewed") -- this function doesn't
    check that itself, it only clears fields.
    """
    visit.model_prediction = None
    visit.model_confidence = None
    visit.svm_prediction = None
    visit.svm_confidence = None
    visit.qsvm_prediction = None
    visit.qsvm_confidence = None


def check_and_run_prediction(db: Session, visit: Visit) -> bool:
    """If the screening visit is data-complete, run the model and advance it.

    Returns True if a prediction was produced (visit moved to pending_review),
    False if the visit isn't ready yet. Safe to call after each modality upload.
    The caller is expected to commit; this only mutates the ``visit`` in-session.
    """
    if visit.visit_type != "screening":
        return False
    if not (visit.mri_status == "done" and visit.speech_status == "done"):
        return False
    if visit.model_prediction is not None:
        # Already predicted — don't re-run or clobber a saved diagnosis flow.
        return False

    if USE_REAL_MODEL:
        try:
            result = _run_real_pipeline(visit)
        except Exception:  # noqa: BLE001 - a model failure must not lose the upload
            # The uploaded features are already persisted by the caller; leaving
            # the visit in `awaiting_uploads` means it can be retried once the
            # model problem is fixed, rather than silently showing a fake result.
            logger.exception("Real pipeline failed for visit %s", visit.id)
            return False
    else:
        result = _placeholder_prediction(visit)

    # Classical SVM result — the one agreement_flag compares the doctor against
    # (see the 2026-09-25 note at the top of this file for why it's the SVM and
    # not the QSVM). svm_prediction/svm_confidence are no longer written.
    visit.model_prediction, visit.model_confidence = result
    visit.status = "pending_review"

    # Research view: the QSVM on the same row. Never blocks the canonical result.
    if USE_REAL_MODEL:
        visit.qsvm_prediction, visit.qsvm_confidence = _run_qsvm(visit) or (None, None)
    else:
        visit.qsvm_prediction, visit.qsvm_confidence = result
    return True


def _run_qsvm(visit: Visit) -> tuple[str, float] | None:
    qsvm = _load_qsvm()
    if qsvm is None:
        return None
    try:
        started = time.perf_counter()
        result = _predict_with(qsvm, build_feature_row(visit))
        logger.info(
            "visit=%s qsvm=%s (margin %.4f, %.0f ms)",
            visit.id, result[0], result[1], (time.perf_counter() - started) * 1000,
        )
        return result
    except Exception:  # noqa: BLE001 - research view only
        logger.exception("QSVM failed for visit %s; research view left empty", visit.id)
        return None


def _placeholder_prediction(visit: Visit) -> tuple[str, float]:
    """Deterministic stand-in so the review flow works without the pickle.

    NOT a clinical output. Keyed off the visit id purely so tests are repeatable.
    """
    prediction = "Demented" if (visit.id.int % 2 == 0) else "Nondemented"
    return prediction, 0.5


def _run_real_pipeline(visit: Visit) -> tuple[str, float]:
    """Real fused inference with the classical SVM: returns (label, margin)."""
    svm = _load_models()
    row = build_feature_row(visit)

    result = _predict_with(svm, row)

    logger.info("visit=%s svm=%s (margin %.4f)", visit.id, result[0], result[1])
    return result
