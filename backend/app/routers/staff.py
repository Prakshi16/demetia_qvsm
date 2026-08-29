"""Staff management (§5) — the hospital_admin's dashboard.

An admin manages people, never patients: this router lists the clinicians and
receptionists at the admin's own hospital and adds new ones. It is the
authenticated, hospital-scoped counterpart to the public
``POST /auth/register-staff`` self-signup — same effect, but the admin stays
logged in as themselves and the new account lands in *their* hospital, never one
passed in the body.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import CurrentUser, require_hospital_admin
from app.models import User
from app.schemas import StaffCreate, StaffListItem
from app.security import hash_password

router = APIRouter(prefix="/staff", tags=["staff"])


@router.get("", response_model=list[StaffListItem])
def list_staff(
    db: Session = Depends(get_db),
    admin: CurrentUser = Depends(require_hospital_admin),
) -> list[User]:
    """Everyone at the admin's hospital, admins included, newest role-grouped."""
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
    """Add a clinician or receptionist to the admin's own hospital."""
    user = User(
        hospital_id=admin.hospital_id,
        name=body.name,
        email=body.email,
        password_hash=hash_password(body.password),
        role=body.role,  # schema restricts to receptionist/clinician
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )
    db.refresh(user)
    return user
