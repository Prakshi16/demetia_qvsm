"""The caller's own hospital — identity, editable profile, and logo (fault #4).

``GET /hospital`` is open to any signed-in user (the nav bar and profile screen
read it); edits and the logo upload are ``hospital_admin`` only.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from supabase import create_client

from app.config import settings
from app.db import get_db
from app.deps import CurrentUser, get_current_user, require_hospital_admin
from app.models import Hospital
from app.schemas import HospitalDetailOut, HospitalUpdate
from app.services.hospitals import assert_hospital_name_available

router = APIRouter(prefix="/hospital", tags=["hospital"])

ALLOWED_LOGO_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".svg")
MAX_LOGO_SIZE_BYTES = 2 * 1024 * 1024


def _load(db: Session, user: CurrentUser) -> Hospital:
    hospital = db.get(Hospital, user.hospital_id)
    if hospital is None:  # a valid token whose hospital was deleted
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Hospital not found"
        )
    return hospital


@router.get("", response_model=HospitalDetailOut)
def get_hospital(
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> Hospital:
    return _load(db, user)


@router.patch("", response_model=HospitalDetailOut)
def update_hospital(
    body: HospitalUpdate,
    db: Session = Depends(get_db),
    admin: CurrentUser = Depends(require_hospital_admin),
) -> Hospital:
    hospital = _load(db, admin)
    fields = body.model_dump(exclude_unset=True)

    new_name = fields.get("name", hospital.name)
    new_pincode = fields.get("pincode", hospital.pincode)
    if "name" in fields or "pincode" in fields:
        assert_hospital_name_available(
            db, name=new_name, pincode=new_pincode, exclude_id=hospital.id
        )

    for key, value in fields.items():
        setattr(hospital, key, value)
    db.commit()
    db.refresh(hospital)
    return hospital


@router.post("/logo", response_model=HospitalDetailOut)
async def upload_logo(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin: CurrentUser = Depends(require_hospital_admin),
) -> Hospital:
    hospital = _load(db, admin)

    extension = os.path.splitext((file.filename or "").lower())[1]
    if extension not in ALLOWED_LOGO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type. Use: {', '.join(ALLOWED_LOGO_EXTENSIONS)}.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Logo file is empty."
        )
    if len(file_bytes) > MAX_LOGO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Logo exceeds the 2 MB limit.",
        )

    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    bucket = client.storage.from_(settings.SUPABASE_PUBLIC_BUCKET)
    storage_path = f"hospital-logos/{hospital.id}/logo{extension}"
    try:
        bucket.upload(
            storage_path,
            file_bytes,
            {
                "content-type": file.content_type or "application/octet-stream",
                "upsert": "true",
            },
        )
    except Exception as exc:  # noqa: BLE001 - most likely the public bucket is missing
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Could not store the logo. Check that a public storage bucket "
                f"named '{settings.SUPABASE_PUBLIC_BUCKET}' exists."
            ),
        ) from exc

    public_url = bucket.get_public_url(storage_path)
    # get_public_url appends a cache-busting query on some client versions; keep
    # whatever it returns but strip a trailing "?" with no params.
    hospital.logo_url = public_url.rstrip("?")
    db.commit()
    db.refresh(hospital)
    return hospital
