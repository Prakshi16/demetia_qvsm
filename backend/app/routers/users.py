"""Staff management — ``hospital_admin`` only (fault #5).

An admin provisions receptionist/clinician accounts with a temporary password;
the new account carries ``must_change_password`` until the user sets their own via
``POST /auth/change-password``. Everything here is scoped to the admin's own
hospital.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import CurrentUser, require_admin
from app.models import User
from app.schemas import ResetPasswordRequest, StaffCreate, StaffOut
from app.security import hash_password
from app.services.audit import record_audit

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[StaffOut])
def list_staff(
    db: Session = Depends(get_db),
    admin: CurrentUser = Depends(require_admin),
) -> list[User]:
    """Every account at the admin's hospital, the admin included."""
    return (
        db.query(User)
        .filter(User.hospital_id == admin.hospital_id)
        .order_by(User.name)
        .all()
    )


@router.post("", response_model=StaffOut, status_code=status.HTTP_201_CREATED)
def create_staff(
    body: StaffCreate,
    db: Session = Depends(get_db),
    admin: CurrentUser = Depends(require_admin),
) -> User:
    """Provision a receptionist or clinician with a temporary password."""
    user = User(
        hospital_id=admin.hospital_id,
        name=body.name,
        email=body.email,
        password_hash=hash_password(body.temporary_password),
        role=body.role,
        must_change_password=True,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        )
    record_audit(
        db,
        hospital_id=admin.hospital_id,
        user_id=admin.user_id,
        action="create_staff",
        target_type="user",
        target_id=user.id,
    )
    db.commit()
    db.refresh(user)
    return user


@router.post("/{user_id}/reset-password", response_model=StaffOut)
def reset_staff_password(
    user_id: uuid.UUID,
    body: ResetPasswordRequest,
    db: Session = Depends(get_db),
    admin: CurrentUser = Depends(require_admin),
) -> User:
    """Set a new temporary password for a staff member and force a change."""
    if user_id == admin.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use change-password for your own account",
        )

    user = (
        db.query(User)
        .filter(User.id == user_id, User.hospital_id == admin.hospital_id)
        .first()
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Staff member not found"
        )

    user.password_hash = hash_password(body.new_password)
    user.must_change_password = True
    record_audit(
        db,
        hospital_id=admin.hospital_id,
        user_id=admin.user_id,
        action="reset_staff_password",
        target_type="user",
        target_id=user.id,
    )
    db.commit()
    db.refresh(user)
    return user
