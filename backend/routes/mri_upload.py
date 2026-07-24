from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Path, UploadFile, status

from backend.services.mri_features import (
    MriFeatureExtractionNotReady,
    extract_mri_features,
    validate_mri_feature_vector,
)


router = APIRouter(tags=["MRI Upload"])

ALLOWED_MRI_EXTENSIONS = (".nii", ".nii.gz", ".dcm", ".dicom", ".mgh", ".mgz")
MAX_UPLOAD_SIZE_BYTES = 250 * 1024 * 1024


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


@router.post("/api/v1/visits/{id}/mri-upload")
async def upload_mri(
    id: str = Path(..., min_length=1, description="Visit ID"),
    file: UploadFile = File(...),
) -> dict[str, object]:
    _validate_upload(file)
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
    finally:
        await file.close()

    # TODO:
    # Store `features` in visits.mri_feature_vector when Prakshi's Visit model and
    # database service are available.

    # TODO:
    # Set visits.mri_status = 'done' when Prakshi's Visit model and database service
    # are available.

    # TODO:
    # Ask Prakshi's prediction-trigger glue to run if both mri_status and
    # speech_status are done.

    return {
        "status": "success",
        "features": features,
        "mri_status": "done",
    }
