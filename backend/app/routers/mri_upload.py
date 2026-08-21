"""
MRI upload endpoint.

Uploads an MRI scan, stores it in Supabase Storage, extracts the 4-feature
MRI vector, saves it to the visit, and triggers the prediction seam if both
modalities are complete.

Mirrors the structure of routers/speech.py deliberately — both upload
endpoints should look the same so the visit flow behaves identically for
either modality.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from supabase import create_client

from app.config import settings
from app.db import get_db
from app.deps import CurrentUser, get_current_user, get_scoped_query
from app.models import Visit
from app.schemas import VisitDetailOut
from app.services.mri_features import (
    MriFeatureExtractionNotReady,
    extract_mri_features,
    validate_mri_feature_vector,
)
from app.services.prediction import check_and_run_prediction

router = APIRouter(prefix="/visits", tags=["MRI Upload"])

ALLOWED_MRI_EXTENSIONS = (".nii", ".nii.gz", ".dcm", ".dicom", ".mgh", ".mgz")

# 50 MB — the Supabase free-tier per-file storage limit. Anything larger would
# pass validation here and then fail at the Storage upload, which is a much
# more confusing error to debug.
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024


def _has_allowed_extension(filename: str) -> bool:
    normalized_filename = filename.lower()
    return any(normalized_filename.endswith(extension) for extension in ALLOWED_MRI_EXTENSIONS)


def _get_upload_size(file: UploadFile) -> int:
    current_position = file.file.tell()
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(current_position)
    return size


def _validate_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MRI file is required.",
        )

    if not _has_allowed_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported MRI file type. Accepted formats: {', '.join(ALLOWED_MRI_EXTENSIONS)}.",
        )

    upload_size = _get_upload_size(file)

    if upload_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MRI file cannot be empty.",
        )

    if upload_size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="MRI file exceeds the maximum upload size.",
        )


@router.post(
    "/{visit_id}/mri-upload",
    response_model=VisitDetailOut,
)
async def upload_mri(
    visit_id: uuid.UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> Visit:
    """Upload an MRI scan for a screening visit."""

    # ------------------------------------------------------------------
    # Load visit (hospital scoped — Rule 12, never use a raw db.query)
    # ------------------------------------------------------------------
    visit = (
        get_scoped_query(db, Visit, user)
        .filter(Visit.id == visit_id)
        .first()
    )

    if visit is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Visit not found",
        )

    if visit.visit_type != "screening":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MRI upload is only allowed for screening visits",
        )

    # ------------------------------------------------------------------
    # Validate the uploaded file
    # ------------------------------------------------------------------
    _validate_upload(file)

    # ------------------------------------------------------------------
    # Read uploaded file
    # ------------------------------------------------------------------
    file_bytes = await file.read()

    # ------------------------------------------------------------------
    # Upload the raw scan to Supabase Storage
    # ------------------------------------------------------------------
    client = create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_KEY,
    )

    storage_path = f"mri/{visit_id}/{file.filename}"

    client.storage.from_(settings.SUPABASE_BUCKET).upload(
        storage_path,
        file_bytes,
        {
            "content-type": file.content_type
            or "application/octet-stream",
            "upsert": "true",
        },
    )

    # ------------------------------------------------------------------
    # Extract MRI features (unchanged — services/mri_features.py)
    # ------------------------------------------------------------------
    file.file.seek(0)

    try:
        features = validate_mri_feature_vector(extract_mri_features(file))
    except MriFeatureExtractionNotReady as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # ------------------------------------------------------------------
    # Save to database
    # ------------------------------------------------------------------
    visit.mri_feature_vector = features
    visit.mri_status = "done"

    # ------------------------------------------------------------------
    # Trigger prediction (runs the model only once BOTH modalities are done)
    # ------------------------------------------------------------------
    check_and_run_prediction(db, visit)

    db.commit()
    db.refresh(visit)

    return visit
