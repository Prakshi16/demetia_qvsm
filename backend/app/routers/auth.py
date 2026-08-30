"""Auth + public hospital list (§5).

``register-hospital`` and ``login`` return a JWT so the client is logged in
immediately. Staff accounts are no longer self-service — a ``hospital_admin``
provisions them via ``POST /staff`` (see routers/staff.py) and the new staff
member is forced through ``POST /auth/change-password`` on first sign-in.

Login addresses are never typed: ``register-hospital`` takes an ``email_domain``
and the admin's own address is derived from their name (services/emails.py), as
every staff address then is.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import CurrentUser, get_current_user
from app.models import Hospital, User
from app.schemas import (
    ChangePasswordRequest,
    HospitalOut,
    LoginRequest,
    RegisterHospitalRequest,
    TokenResponse,
    UserOut,
)
from app.security import create_access_token, hash_password, verify_password
from app.services.emails import allocate_email
from app.services.hospitals import assert_hospital_name_available

router = APIRouter(tags=["auth"])


def _token_for(user: User) -> TokenResponse:
    token = create_access_token(
        user_id=user.id,
        hospital_id=user.hospital_id,
        role=user.role,
        name=user.name,
        must_change_password=user.must_change_password,
    )
    return TokenResponse(token=token, user=UserOut.model_validate(user))


@router.get("/hospitals", response_model=list[HospitalOut])
def list_hospitals(db: Session = Depends(get_db)) -> list[Hospital]:
    """Public. Kept for an 'is my hospital already registered?' check — sign-up
    no longer uses a picker (staff can't self-join)."""
    return db.query(Hospital).order_by(Hospital.name).all()


@router.post("/auth/register-hospital", response_model=TokenResponse)
def register_hospital(
    body: RegisterHospitalRequest, db: Session = Depends(get_db)
) -> TokenResponse:
    """Create a hospital and its first user (a ``hospital_admin``)."""
    assert_hospital_name_available(
        db, name=body.hospital_name, pincode=body.pincode
    )

    hospital = Hospital(
        name=body.hospital_name,
        address=body.address,
        pincode=body.pincode,
        city=body.city,
        email_domain=body.email_domain,
    )
    db.add(hospital)
    db.flush()  # assign hospital.id before creating the admin user

    # The admin's login address is derived from their name + the chosen domain,
    # exactly like every staff account will be (services/emails.py).
    admin = User(
        hospital_id=hospital.id,
        name=body.admin_name,
        email=allocate_email(db, name=body.admin_name, domain=body.email_domain),
        password_hash=hash_password(body.password),
        role="hospital_admin",
        must_change_password=False,
    )
    db.add(admin)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )
    db.refresh(admin)
    return _token_for(admin)


@router.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Email + password. Role is read from the record, never chosen at sign-in."""
    # Stored addresses are lowercase (services/emails.py); match case-insensitively
    # so a capitalised sign-in still works.
    email = (body.email or "").strip().lower()
    user = db.query(User).filter(func.lower(User.email) == email).first()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    return _token_for(user)


@router.post("/auth/change-password", response_model=TokenResponse)
def change_password(
    body: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current: CurrentUser = Depends(get_current_user),
) -> TokenResponse:
    """Change your own password. Clears ``must_change_password`` and returns a
    fresh token so the client's forced-change gate lifts."""
    user = db.get(User, current.user_id)
    if user is None:  # token valid but the account is gone
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )

    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    if verify_password(body.new_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from the current one",
        )

    user.password_hash = hash_password(body.new_password)
    user.must_change_password = False
    db.commit()
    db.refresh(user)
    return _token_for(user)
