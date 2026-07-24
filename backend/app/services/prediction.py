"""Model-serving seam for the fused QSVM prediction.

``check_and_run_prediction`` is the exact function Bishal's MRI upload and
Sheetal's speech upload endpoints call at the very end of their handlers, once
they've stored their feature vector and set their modality status to ``done``.
When BOTH modalities are done it runs the fused pipeline and moves the visit into
the clinician's queue (``status='pending_review'``).

=============================================================================
PLACEHOLDER — the real model glue is the immediately-following pass, not this
branch. The real ``_run_real_pipeline`` must:
  * load results/qsvm_model.pkl once at app startup (not per request),
  * assemble the 27-feature row in the exact Phase-1 order
    (clinical [CDR, MMSE, ASF, EDUC, SES] -> mri[4] -> speech[18]),
  * derive ASF from eTIV (OASIS ASF ~= const/eTIV; fit const on the synthetic
    CSV and LOG THE R^2 as an approximation — wanted for the report),
  * take confidence from decision_function (QSVC has no predict_proba), labelled
    a margin, not a probability,
  * NEVER apply SIGMA_FRAC noise at inference,
  * be timed once locally (quantum kernel vs every support vector can be slow;
    if slow, switch this to an async/polling design).
Until then, the stub below sets a deterministic placeholder so the downstream
pending_review -> diagnosis -> agreement_flag flow is fully testable.
=============================================================================
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Visit

# Flip to True only in the model-serving pass, once _run_real_pipeline exists.
USE_REAL_MODEL = False


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
        prediction, confidence = _run_real_pipeline(visit)
    else:
        prediction, confidence = _placeholder_prediction(visit)

    visit.model_prediction = prediction
    visit.model_confidence = confidence
    visit.status = "pending_review"
    return True


def _placeholder_prediction(visit: Visit) -> tuple[str, float]:
    """Deterministic stand-in so the review flow works before the real pipeline.

    NOT a clinical output. Keyed off the visit id purely so tests are repeatable.
    """
    prediction = "Demented" if (visit.id.int % 2 == 0) else "Nondemented"
    return prediction, 0.5


def _run_real_pipeline(visit: Visit) -> tuple[str, float]:  # pragma: no cover
    """Real fused QSVM inference — implemented in the model-serving pass."""
    raise NotImplementedError(
        "Real QSVM serving lands in the next pass; USE_REAL_MODEL must stay False "
        "until results/qsvm_model.pkl loading + ASF derivation are wired in."
    )
