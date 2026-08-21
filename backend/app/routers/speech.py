"""
Speech upload endpoint.

Uploads a speech recording, stores it in Supabase Storage,
extracts the 18-feature speech vector, saves it to the visit,
and triggers prediction if both modalities are complete.
"""

from __future__ import annotations

import io
import os
import subprocess
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from supabase import create_client

from app.config import settings
from app.db import get_db
from app.deps import CurrentUser, get_current_user, get_scoped_query
from app.models import Visit
from app.schemas import VisitDetailOut
from app.services.prediction import check_and_run_prediction
from app.services.speech_features import extract_speech_features

router = APIRouter(prefix="/visits", tags=["speech"])

# .webm and .m4a are here because <SpeechCapture /> records through
# MediaRecorder, which only ever produces one of those two — without them the
# record button (the whole point of that component) 400s before the audio is
# ever looked at. Decoding them needs ffmpeg in the image; see backend/Dockerfile.
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm"}


# librosa reads whatever libsndfile reads — wav and mp3, but not the opus and
# aac that MediaRecorder produces. librosa 0.11 also dropped the old audioread
# fallback, so it will not reach for ffmpeg on its own: it raises
# "Format not recognised" and the upload 500s. Converting up front keeps that
# decision here, in the upload layer, and leaves the extractor with the single
# wav-shaped input it expects.
TRANSCODE_EXTENSIONS = {".m4a", ".webm"}


def _to_wav(audio_bytes: bytes) -> bytes:
    """Decode a browser recording to 16-bit mono PCM wav via ffmpeg."""
    try:
        completed = subprocess.run(
            [
                "ffmpeg", "-loglevel", "error",
                "-i", "pipe:0",
                "-ac", "1",              # mono: the extractor averages anyway
                "-c:a", "pcm_s16le",
                "-f", "wav", "pipe:1",
            ],
            input=audio_bytes,
            capture_output=True,
            timeout=60,
            check=True,
        )
    except FileNotFoundError as exc:  # ffmpeg missing from the image
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audio conversion is unavailable on this server.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio file took too long to decode.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio file could not be decoded.",
        ) from exc

    return completed.stdout


@router.post(
    "/{visit_id}/speech-upload",
    response_model=VisitDetailOut,
)
async def upload_speech(
    visit_id: uuid.UUID,
    file: UploadFile,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> Visit:
    """
    Upload a speech recording for a screening visit.
    """

    # ------------------------------------------------------------------
    # Load visit (hospital scoped)
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
            detail="Speech upload is only allowed for screening visits",
        )

    # ------------------------------------------------------------------
    # Validate extension
    # ------------------------------------------------------------------
    filename = file.filename or ""

    extension = os.path.splitext(filename.lower())[1]

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio file type. Accepted formats: "
                   f"{', '.join(sorted(ALLOWED_EXTENSIONS))}.",
        )

    # ------------------------------------------------------------------
    # Read uploaded file
    # ------------------------------------------------------------------
    file_bytes = await file.read()

    # ------------------------------------------------------------------
    # Upload to Supabase Storage
    # ------------------------------------------------------------------
    client = create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_KEY,
    )

    storage_path = f"speech/{visit_id}/{filename}"

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
    # Extract speech features
    # ------------------------------------------------------------------
    audio_bytes = (
        _to_wav(file_bytes)
        if extension in TRANSCODE_EXTENSIONS
        else file_bytes
    )

    features = extract_speech_features(io.BytesIO(audio_bytes))

    if len(features) != 18:
        raise HTTPException(
            status_code=500,
            detail="Speech feature extraction failed",
        )

    # ------------------------------------------------------------------
    # Save to database
    # ------------------------------------------------------------------
    visit.speech_feature_vector = features
    visit.speech_status = "done"

    # ------------------------------------------------------------------
    # Trigger prediction
    # ------------------------------------------------------------------
    check_and_run_prediction(db, visit)

    db.commit()
    db.refresh(visit)

    return visit