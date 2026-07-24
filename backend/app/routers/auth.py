"""Auth + public hospital list (§5). All three auth endpoints return a JWT so the
client is logged in immediately after register/login (no separate login step).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Hospital, User
from app.schemas import (
    HospitalOut,
    LoginRequest,
    RegisterHospitalRequest,
    RegisterStaffRequest,
    TokenResponse,
    UserOut,
)
from app.security import create_access_token, hash_password, verify_password

router = APIRouter(tags=["auth"])


def _token_for(user: User) -> TokenResponse:
    token = create_access_token(
        user_id=user.id,
        hospital_id=user.hospital_id,
        role=user.role,
        name=user.name,
    )
    return TokenResponse(token=token, user=UserOut.model_validate(user))


@router.get("/hospitals", response_model=list[HospitalOut])
def list_hospitals(db: Session = Depends(get_db)) -> list[Hospital]:
    """Public — powers the sign-up hospital picker. No auth required."""
    return db.query(Hospital).order_by(Hospital.name).all()


@router.post("/auth/register-hospital", response_model=TokenResponse)
def register_hospital(
    body: RegisterHospitalRequest, db: Session = Depends(get_db)
) -> TokenResponse:
    """Create a hospital and its first user (a ``hospital_admin``)."""
    hospital = Hospital(name=body.hospital_name, address=body.address)
    db.add(hospital)
    db.flush()  # assign hospital.id before creating the admin user

    admin = User(
        hospital_id=hospital.id,
        name=body.admin_name,
        email=body.admin_email,
        password_hash=hash_password(body.password),
        role="hospital_admin",
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


@router.post("/auth/register-staff", response_model=TokenResponse)
def register_staff(
    body: RegisterStaffRequest, db: Session = Depends(get_db)
) -> TokenResponse:
    """Self-service staff sign-up under an existing hospital (no approval step)."""
    hospital = db.get(Hospital, body.hospital_id)
    if hospital is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Hospital not found"
        )

    user = User(
        hospital_id=body.hospital_id,
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
    return _token_for(user)


@router.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    """Email + password. Role is read from the record, never chosen at sign-in."""
    user = db.query(User).filter(User.email == body.email).first()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    return _token_for(user)
