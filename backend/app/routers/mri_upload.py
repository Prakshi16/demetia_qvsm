"""
MRI upload endpoint.

Uploads an MRI scan, stores it in Supabase Storage, extracts the 2-feature
MRI vector, saves it to the visit, and triggers prediction if both modalities
are complete.

Accepts one file (``file``) or several (``files``). Several files arrive when a
scan can't travel as one file: an Analyze ``.hdr`` + ``.img`` pair, or a DICOM
series. They are bundled into one ``.zip`` in storage so the visit still has
exactly one MRI object. Every accepted format is listed in
``services/mri_io.py``.

Re-upload is allowed (e.g. the receptionist picked the wrong file) any time before a
diagnosis has been saved -- this simply overwrites the stored feature vector; if a
prediction had already fired on the old (wrong) file, it's cleared and recomputed from
the corrected one. Once the visit is `reviewed`, re-upload is refused outright.
"""

from __future__ import annotations

import io
import re
import uuid
import zipfile
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from supabase import create_client

from app.config import settings
from app.db import get_db
from app.deps import CurrentUser, get_current_user, get_scoped_query
from app.models import Visit
from app.schemas import VisitDetailOut
from app.services.mri_features import (
    MriFeatureExtractionError,
    MriFeatureExtractionNotReady,
    extract_from_files,
    validate_mri_feature_vector,
)
from app.services.mri_io import ACCEPTED_EXTENSIONS, has_accepted_extension
from app.services.prediction import check_and_run_prediction, clear_stale_prediction

router = APIRouter(prefix="/visits", tags=["MRI Upload"])

# Kept for backwards-compatible imports; the full list lives in mri_io.
ALLOWED_MRI_EXTENSIONS = ACCEPTED_EXTENSIONS
# The Supabase free tier's per-object limit; one stored object per visit.
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024
MAX_FILES = 2000
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

UNSUPPORTED_MESSAGE = (
    "Unsupported MRI file type. Accepted: NIfTI (.nii, .nii.gz), MGH (.mgh, .mgz), "
    "Analyze (.hdr + .img), DICOM (.dcm, a series, or a .zip), or a slice image "
    "(.png, .jpg, .bmp, .tif)."
)


def _safe_name(filename: str) -> str:
    """A storage-key-safe basename that keeps the (compound) extension."""
    base = PurePosixPath((filename or "scan").replace("\\", "/")).name
    cleaned = _SAFE_CHARS.sub("_", base).strip("._") or "scan"
    return cleaned[-120:]


async def _read_uploads(uploads: list[UploadFile]) -> list[tuple[str, bytes]]:
    if not uploads:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="MRI file is required.")
    if len(uploads) > MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files in one upload (limit {MAX_FILES}).",
        )

    files: list[tuple[str, bytes]] = []
    total = 0
    for upload in uploads:
        name = upload.filename or ""
        if not name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="MRI file is required.")
        raw = await upload.read()
        total += len(raw)
        if total > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="MRI upload exceeds the 50 MB limit. Upload a compressed .nii.gz, or zip the series.",
            )
        files.append((name, raw))

    if all(len(raw) == 0 for _, raw in files):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="MRI file cannot be empty.")

    # A single file must name its format; inside a multi-file DICOM series,
    # extension-less slices are fine (they're recognised by their preamble).
    if len(files) == 1 and not has_accepted_extension(files[0][0]):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=UNSUPPORTED_MESSAGE)
    return files


def _storage_object(files: list[tuple[str, bytes]]) -> tuple[str, bytes, str]:
    """(object name, bytes, content type) for what goes into storage."""
    if len(files) == 1:
        name, raw = files[0]
        return _safe_name(name), raw, "application/octet-stream"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (name, raw) in enumerate(files):
            archive.writestr(f"{index:04d}_{_safe_name(name)}", raw)
    stem = _safe_name(files[0][0]).split(".")[0] or "scan"
    return f"{stem}_bundle.zip", buffer.getvalue(), "application/zip"


@router.post(
    "/{visit_id}/mri-upload",
    response_model=VisitDetailOut,
)
async def upload_mri(
    visit_id: uuid.UUID,
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> Visit:
    """Upload an MRI scan (one file, or several that together form one scan)."""

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

    if visit.status == "reviewed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Diagnosis already saved for this visit; re-upload is not allowed.",
        )

    uploads = ([file] if file is not None else []) + list(files or [])
    payload = await _read_uploads(uploads)

    # Extract before storing: a file that can't be read as a scan is rejected
    # without leaving an orphan object in the bucket. CPU-bound, so it runs off
    # the event loop.
    try:
        extraction = await run_in_threadpool(extract_from_files, payload)
        features = validate_mri_feature_vector(extraction.features)
    except MriFeatureExtractionNotReady as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc
    except (MriFeatureExtractionError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    object_name, object_bytes, content_type = _storage_object(payload)
    storage_path = f"mri/{visit_id}/{object_name}"

    client = create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_KEY,
    )
    try:
        client.storage.from_(settings.SUPABASE_BUCKET).upload(
            storage_path,
            object_bytes,
            {"content-type": content_type, "upsert": "true"},
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The scan was read, but saving it to storage failed. Try again.",
        ) from exc

    visit.mri_feature_vector = features
    visit.mri_extraction = extraction.meta
    visit.mri_object_path = storage_path
    visit.mri_status = "done"

    if visit.model_prediction is not None:
        # Re-upload after a prediction already fired (the receptionist picked the
        # wrong file the first time): clear it so it recomputes from this MRI.
        clear_stale_prediction(visit)

    await run_in_threadpool(check_and_run_prediction, db, visit)

    db.commit()
    db.refresh(visit)

    return visit
