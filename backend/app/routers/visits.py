"""Visit endpoints (§5): create, detail, and the clinician diagnosis save.

Upload endpoints (/mri-upload, /speech-upload) are intentionally NOT here —
they're Bishal's and Sheetal's, and they call
``services.prediction.check_and_run_prediction`` once their modality is done.
"""
from __future__ import annotations

import mimetypes
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session
from supabase import create_client

from app.config import settings
from app.db import get_db
from app.deps import (
    CurrentUser,
    get_current_user,
    get_scoped_query,
    require_clinician,
    require_receptionist,
)
from app.models import DiagnosisHistory, Patient, Visit
from app.schemas import DiagnosisCreate, VisitCreate, VisitDetailOut, VisitFileOut
from app.services.audio_view import PLAYER_FILENAME, PLAYER_MEDIA_TYPE, to_playable_mp3
from app.services.audit import record_audit
from app.services.scan_view import VIEWER_FILENAME, VIEWER_MEDIA_TYPE, to_viewer_nifti
from app.services.visit_logic import decide_visit_type

router = APIRouter(prefix="/visits", tags=["visits"])


def _load_scoped_visit(db: Session, user: CurrentUser, visit_id: uuid.UUID) -> Visit:
    visit = get_scoped_query(db, Visit, user).filter(Visit.id == visit_id).first()
    if visit is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Visit not found"
        )
    return visit


@router.post("", response_model=VisitDetailOut, status_code=status.HTTP_201_CREATED)
def create_visit(
    body: VisitCreate,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_receptionist),
) -> Visit:
    """Create a screening or follow-up visit.

    Receptionist-only: the front desk enters all clinical data and uploads;
    the clinician's job is limited to reviewing the result.

    follow_up  -> completed immediately, no review, no model.
    screening  -> awaiting_uploads; modalities added later via the upload endpoints.

    A follow_up is only accepted when §4 actually permits it for this patient
    (confirmed diagnosis on record, within the re-screen window) — otherwise 400.
    Screening is always allowed (it's the safe default / manual override).
    """
    patient = (
        get_scoped_query(db, Patient, user)
        .filter(Patient.id == body.patient_id)
        .first()
    )
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
        )

    if body.visit_type == "follow_up":
        last_screening = (
            get_scoped_query(db, Visit, user)
            .filter(
                Visit.patient_id == body.patient_id,
                Visit.visit_type == "screening",
            )
            .order_by(Visit.visit_date.desc())
            .first()
        )
        allowed, reason = decide_visit_type(last_screening, force_screening=False)
        if allowed != "follow_up":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Follow-up not allowed: {reason}",
            )
        visit = Visit(
            patient_id=body.patient_id,
            hospital_id=user.hospital_id,
            visit_type="follow_up",
            created_by_user_id=user.user_id,
            mmse=body.mmse,
            cdr=body.cdr,
            mri_status="not_applicable",
            speech_status="not_applicable",
            requires_review=False,
            status="completed",
        )
    else:  # screening
        visit = Visit(
            patient_id=body.patient_id,
            hospital_id=user.hospital_id,
            visit_type="screening",
            created_by_user_id=user.user_id,
            mmse=body.mmse,
            cdr=body.cdr,
            edu=body.edu,
            ses=body.ses,
            mri_status="idle",
            speech_status="idle",
            requires_review=True,
            status="awaiting_uploads",
        )

    db.add(visit)
    db.commit()
    db.refresh(visit)
    return visit


@router.get("/{visit_id}", response_model=VisitDetailOut)
def get_visit(
    visit_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> Visit:
    """Full visit detail incl. the diagnosis_history list (same-day revisions)."""
    return _load_scoped_visit(db, user, visit_id)


_FILE_KINDS = {
    "mri": ("mri_status", "mri_object_path", True),
    "speech": ("speech_status", "speech_object_path", False),
}


def _uploads_bucket():
    return create_client(
        settings.SUPABASE_URL, settings.SUPABASE_KEY
    ).storage.from_(settings.SUPABASE_BUCKET)


def _resolve_object_path(bucket, visit: Visit, kind: str) -> str:
    """The storage key for a visit's ``kind`` upload, or 404.

    Falls back to listing the folder for rows uploaded before migration 005
    recorded the path.
    """
    status_attr, path_attr, _ = _FILE_KINDS[kind]
    if getattr(visit, status_attr) != "done":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {kind} file for this visit",
        )
    object_path = getattr(visit, path_attr)
    if object_path:
        return object_path
    try:
        entries = bucket.list(f"{kind}/{visit.id}")
    except Exception:  # noqa: BLE001 - any storage error means "not found"
        entries = []
    names = [e["name"] for e in entries if e.get("name")]
    if not names:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{kind} file not found in storage",
        )
    return f"{kind}/{visit.id}/{names[0]}"


@router.get("/{visit_id}/file/{kind}", response_model=VisitFileOut)
def get_visit_file(
    visit_id: uuid.UUID,
    kind: str,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> VisitFileOut:
    """A short-lived signed URL for a visit's raw MRI scan or speech recording.

    Any signed-in user at the visit's hospital may fetch it — the clinician
    needs it to review, the receptionist to confirm the upload. The bucket is
    private, so the URL is minted per request and expires in an hour. This
    returns the file *as uploaded* (e.g. a `.dcm`); the in-browser viewer uses
    ``/scan`` instead, which normalises to NIfTI.
    """
    if kind not in _FILE_KINDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Unknown file kind"
        )
    _, _, force_download = _FILE_KINDS[kind]

    visit = _load_scoped_visit(db, user, visit_id)
    bucket = _uploads_bucket()
    object_path = _resolve_object_path(bucket, visit, kind)

    filename = object_path.rsplit("/", 1)[-1]
    options = {"download": filename} if force_download else None
    try:
        signed = (
            bucket.create_signed_url(object_path, 3600, options)
            if options is not None
            else bucket.create_signed_url(object_path, 3600)
        )
        url = signed.get("signedURL") or signed.get("signedUrl")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not generate a download link for this file.",
        ) from exc
    if not url:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Storage did not return a signed URL.",
        )

    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return VisitFileOut(url=url, filename=filename, content_type=content_type)


@router.get("/{visit_id}/scan")
def get_visit_scan(
    visit_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> Response:
    """The MRI scan as gzip-compressed NIfTI, for the in-browser viewer.

    Streamed through the app (not a signed URL) because the browser viewer
    can't attach an auth header, and because a `.dcm` upload has to be decoded
    to NIfTI server-side — Niivue has no DICOM reader.
    """
    visit = _load_scoped_visit(db, user, visit_id)
    bucket = _uploads_bucket()
    object_path = _resolve_object_path(bucket, visit, "mri")
    filename = object_path.rsplit("/", 1)[-1]

    try:
        raw = bucket.download(object_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not read the scan from storage.",
        ) from exc

    try:
        nifti = to_viewer_nifti(raw, filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc

    return Response(
        content=nifti,
        media_type=VIEWER_MEDIA_TYPE,
        headers={"Content-Disposition": f'inline; filename="{VIEWER_FILENAME}"'},
    )


@router.get("/{visit_id}/audio")
def get_visit_audio(
    visit_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> Response:
    """The speech recording as MP3, for the review screen's audio player.

    Transcoded because a browser MediaRecorder produces WebM/Opus, which
    doesn't play in every browser and often has no duration metadata.
    """
    visit = _load_scoped_visit(db, user, visit_id)
    bucket = _uploads_bucket()
    object_path = _resolve_object_path(bucket, visit, "speech")
    filename = object_path.rsplit("/", 1)[-1]

    try:
        raw = bucket.download(object_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not read the recording from storage.",
        ) from exc

    try:
        mp3 = to_playable_mp3(raw, filename)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return Response(
        content=mp3,
        media_type=PLAYER_MEDIA_TYPE,
        headers={"Content-Disposition": f'inline; filename="{PLAYER_FILENAME}"'},
    )


@router.post("/{visit_id}/diagnosis", response_model=VisitDetailOut)
def save_diagnosis(
    visit_id: uuid.UUID,
    body: DiagnosisCreate,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_clinician),
) -> Visit:
    """Save (or same-day-revise) a diagnosis on a screening visit (Rule 5).

    Accepted ONLY when the visit is a screening that is either ``pending_review``
    or ``reviewed`` on the *same UTC calendar day* as its last save. Every other
    state is rejected here at the endpoint, not merely hidden from the dashboard —
    a clinician must not be able to diagnose an incomplete or wrong-day visit by
    POSTing its id directly (insecure-direct-object-reference guard).
    """
    visit = _load_scoped_visit(db, user, visit_id)

    if visit.visit_type != "screening":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only screening visits can be diagnosed",
        )

    now = datetime.now(timezone.utc)
    if visit.status == "pending_review":
        pass  # first diagnosis — allowed
    elif visit.status == "reviewed":
        # same-day revision only. "Same day" = server UTC day boundary (deferred
        # decision): no hospital-local timezone handling in Phase 2.
        last_saved = visit.diagnosis_saved_at
        same_utc_day = (
            last_saved is not None
            and _as_utc(last_saved).date() == now.date()
        )
        if not same_utc_day:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "This visit was reviewed on a previous day; a later revision "
                    "goes through a new follow-up visit, not an edit."
                ),
            )
        # A same-day "revision" that changes nothing is a no-op — reject it
        # rather than write a duplicate history row.
        if (
            body.doctor_diagnosis == visit.doctor_diagnosis
            and (body.doctor_notes or None) == (visit.doctor_notes or None)
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No changes to save.",
            )
    else:
        # awaiting_uploads (model hasn't run) or completed (follow-up) — never valid.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot diagnose a visit in status '{visit.status}'",
        )

    # Append to the audit trail, then overwrite the fast-read mirror on the visit.
    db.add(
        DiagnosisHistory(
            visit_id=visit.id,
            doctor_diagnosis=body.doctor_diagnosis,
            doctor_notes=body.doctor_notes,
            saved_by_user_id=user.user_id,
        )
    )
    visit.doctor_diagnosis = body.doctor_diagnosis
    visit.doctor_notes = body.doctor_notes
    visit.diagnosis_saved_at = now
    visit.diagnosis_saved_by_user_id = user.user_id
    visit.status = "reviewed"
    # Agreement flag (Rule 6): informational only. "Needs further evaluation" is
    # always a mismatch since the model never predicts that class.
    visit.agreement_flag = (
        "match"
        if visit.model_prediction is not None
        and body.doctor_diagnosis == visit.model_prediction
        else "mismatch"
    )

    record_audit(
        db,
        hospital_id=user.hospital_id,
        user_id=user.user_id,
        action="save_diagnosis",
        target_type="visit",
        target_id=visit.id,
    )
    db.commit()
    db.refresh(visit)
    return visit


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
