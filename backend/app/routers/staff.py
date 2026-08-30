"""Staff management (§5) — the hospital_admin's dashboard.

An admin manages people, never patients: this router lists the clinicians and
receptionists at the admin's own hospital, provisions new ones, and resets their
passwords. Staff have no self-service sign-up — an admin creates the account with
a temporary password and the new user is forced to change it on first sign-in
(``must_change_password``). Everything here is scoped to the admin's own
hospital; a hospital_id in the body is never trusted.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import CurrentUser, require_hospital_admin
from app.models import Hospital, User
from app.schemas import ResetPasswordRequest, StaffCreate, StaffListItem
from app.security import hash_password
from app.services.audit import record_audit
from app.services.emails import allocate_email

router = APIRouter(prefix="/staff", tags=["staff"])


@router.get("", response_model=list[StaffListItem])
def list_staff(
    db: Session = Depends(get_db),
    admin: CurrentUser = Depends(require_hospital_admin),
) -> list[User]:
    """Everyone at the admin's hospital, admins included, role-grouped."""
    return (
        db.query(User)
        .filter(User.hospital_id == admin.hospital_id)
        .order_by(User.role, User.created_at)
        .all()
    )


@router.post("", response_model=StaffListItem, status_code=status.HTTP_201_CREATED)
def add_staff(
    body: StaffCreate,
    db: Session = Depends(get_db),
    admin: CurrentUser = Depends(require_hospital_admin),
) -> User:
    """Provision a clinician or receptionist with a temporary password.

    The login address is derived from the name + the hospital's email domain
    (fixed at registration), so the admin never types an email.
    """
    hospital = db.get(Hospital, admin.hospital_id)
    if hospital is None or not hospital.email_domain:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Your hospital has no email domain set — cannot create accounts.",
        )

    user = User(
        hospital_id=admin.hospital_id,
        name=body.name,
        email=allocate_email(db, name=body.name, domain=hospital.email_domain),
        password_hash=hash_password(body.temporary_password),
        role=body.role,  # schema restricts to receptionist/clinician
        must_change_password=True,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
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


@router.post("/{user_id}/reset-password", response_model=StaffListItem)
def reset_staff_password(
    user_id: uuid.UUID,
    body: ResetPasswordRequest,
    db: Session = Depends(get_db),
    admin: CurrentUser = Depends(require_hospital_admin),
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
