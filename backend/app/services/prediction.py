"""Model-serving seam for the fused QSVM prediction.

``check_and_run_prediction`` is the exact function Bishal's MRI upload and
Sheetal's speech upload endpoints call at the very end of their handlers, once
they've stored their feature vector and set their modality status to ``done``.
When BOTH modalities are done it runs the fused pipeline and moves the visit into
the clinician's queue (``status='pending_review'``).

Two predictions are produced: the QSVM ("Quantum SVM") result on
``model_prediction``/``model_confidence`` (the canonical one — agreement_flag
compares the doctor against it), and a classical SVM result on
``svm_prediction``/``svm_confidence``, shown alongside it purely for
research/comparison. The classical SVM feeds no computed field.

=============================================================================
SERVING NOTES (decisions resolved in the model-serving pass)

* **Latency.** A single fused QSVM prediction measured **~445 ms** locally
  (146 support vectors, FidelityStatevectorKernel). That is comfortably inside a
  normal request, so prediction stays SYNCHRONOUS — no async/polling design is
  needed. Both pickles are loaded once at startup, not per request.

* **Confidence is a MARGIN, not a probability.** ``QSVC`` is fit with
  ``probability=False`` and exposes no ``predict_proba``. Both models therefore
  report ``abs(decision_function(...))`` — signed distance from the separating
  hyperplane. It is NOT calibrated and must never be shown to a clinician as a
  percentage likelihood. One ``decision_function`` call also yields the label
  from its sign, so we spend one ~445 ms kernel evaluation per model, not two.

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
from functools import lru_cache
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Visit

logger = logging.getLogger(__name__)

# Flip to False to fall back to the deterministic stub (useful if a teammate is
# working without the pickles present).
USE_REAL_MODEL = True

QSVM_PICKLE = "qsvm_model.pkl"
SVM_PICKLE = "svm_model.pkl"

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
    """Where the Phase 1 pickles live.

    docker-compose mounts ./results at settings.MODEL_DIR (/model). Running
    uvicorn directly outside Docker, that path doesn't exist, so fall back to
    <repo>/results — services/ -> app/ -> backend/ -> repo root.
    """
    configured = Path(settings.MODEL_DIR)
    if (configured / QSVM_PICKLE).exists():
        return configured
    return Path(__file__).resolve().parents[3] / "results"


@lru_cache(maxsize=1)
def _load_models() -> tuple[object, object | None]:
    """Load both pickles once. Cached — never re-read per request.

    The QSVM is required; the classical SVM is optional because it is
    display-only, and a missing comparison model must not take the app down.
    Regenerate it with `python backend/scripts/train_classical_svm.py`.
    """
    directory = _model_dir()

    qsvm_path = directory / QSVM_PICKLE
    with qsvm_path.open("rb") as handle:
        qsvm = pickle.load(handle)
    logger.info("Loaded QSVM pipeline from %s", qsvm_path)

    svm = None
    svm_path = directory / SVM_PICKLE
    if svm_path.exists():
        with svm_path.open("rb") as handle:
            svm = pickle.load(handle)
        logger.info("Loaded classical SVM pipeline from %s", svm_path)
    else:
        logger.warning(
            "Classical SVM pickle missing at %s — the comparison prediction will "
            "be omitted. Run backend/scripts/train_classical_svm.py to create it.",
            svm_path,
        )

    return qsvm, svm


def warm_models() -> bool:
    """Preload the pickles at startup so the first upload isn't slow.

    Returns True if the QSVM loaded. Never raises: a model failure must not stop
    the API from booting — uploads still work, the visit just stays in
    `awaiting_uploads` instead of advancing, and /health reports the problem.
    """
    if not USE_REAL_MODEL:
        return False
    try:
        _load_models()
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
    ("Demented"). Deriving the label from the sign avoids a second ~445 ms kernel
    evaluation that predict() would cost.
    """
    margin = float(pipeline.decision_function([row])[0])
    label = CLASS_LABELS[1] if margin > 0 else CLASS_LABELS[0]
    return label, abs(margin)


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
            qsvm_result, svm_result = _run_real_pipeline(visit)
        except Exception:  # noqa: BLE001 - a model failure must not lose the upload
            # The uploaded features are already persisted by the caller; leaving
            # the visit in `awaiting_uploads` means it can be retried once the
            # model problem is fixed, rather than silently showing a fake result.
            logger.exception("Real pipeline failed for visit %s", visit.id)
            return False
    else:
        qsvm_result = _placeholder_prediction(visit)
        svm_result = qsvm_result

    # QSVM (canonical) result — the one agreement_flag compares the doctor against.
    visit.model_prediction, visit.model_confidence = qsvm_result
    # Classical SVM (display-only comparison). Feeds no computed field.
    if svm_result is not None:
        visit.svm_prediction, visit.svm_confidence = svm_result
    visit.status = "pending_review"
    return True


def _placeholder_prediction(visit: Visit) -> tuple[str, float]:
    """Deterministic stand-in so the review flow works without the pickles.

    NOT a clinical output. Keyed off the visit id purely so tests are repeatable.
    """
    prediction = "Demented" if (visit.id.int % 2 == 0) else "Nondemented"
    return prediction, 0.5


def _run_real_pipeline(
    visit: Visit,
) -> tuple[tuple[str, float], tuple[str, float] | None]:
    """Real fused inference: returns (qsvm_result, classical_svm_result).

    The classical result is None when its pickle is absent — it is display-only,
    so the visit still advances on the QSVM alone.
    """
    qsvm, svm = _load_models()
    row = build_feature_row(visit)

    qsvm_result = _predict_with(qsvm, row)
    svm_result = _predict_with(svm, row) if svm is not None else None

    logger.info(
        "visit=%s qsvm=%s (margin %.4f) svm=%s",
        visit.id,
        qsvm_result[0],
        qsvm_result[1],
        svm_result[0] if svm_result else "unavailable",
    )
    return qsvm_result, svm_result
