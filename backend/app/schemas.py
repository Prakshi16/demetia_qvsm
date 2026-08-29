"""Pydantic request/response models for the §5 API contract.

Response models use ``from_attributes=True`` so they can be built straight from
SQLAlchemy ORM objects. Email fields are plain ``str`` (not ``EmailStr``) to
avoid pulling in ``email-validator`` — format validation isn't required for the
Phase 2 demo and the DB's UNIQUE(email) is the real guard.
"""
from __future__ import annotations

import re
import uuid
from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# Literals mirror the Postgres ENUM value sets in models.py / 001_init.sql.
StaffRole = Literal["receptionist", "clinician"]
ConsentGivenBy = Literal["patient", "guardian"]
VisitType = Literal["screening", "follow_up"]
DoctorDiagnosis = Literal["Nondemented", "Demented", "Needs further evaluation"]

MIN_PASSWORD_LENGTH = 8

# Allowed punctuation in a person's name vs an organisation's name. Neither list
# is about "correctness" — it's a sanity gate so a name can't be digits-only,
# blank, or a wall of symbols. The frontend mirrors these in utils/validate.js.
_PERSON_NAME_EXTRA = set(" -'.")
_ORG_NAME_EXTRA = set(" -'.,&()/#")
_WHITESPACE_RUN = re.compile(r"\s+")


def _clean_name(value: str, *, kind: str, extra: set[str], max_len: int) -> str:
    """Trim, collapse internal whitespace, and reject nonsense (see above)."""
    cleaned = _WHITESPACE_RUN.sub(" ", (value or "").strip())
    if len(cleaned) < 2 or len(cleaned) > max_len:
        raise ValueError(f"{kind} must be between 2 and {max_len} characters")
    if not any(ch.isalpha() for ch in cleaned):
        raise ValueError(f"{kind} must contain at least one letter")
    bad = {ch for ch in cleaned if not (ch.isalpha() or ch.isdigit() or ch in extra)}
    if bad:
        raise ValueError(f"{kind} contains invalid characters: {''.join(sorted(bad))}")
    return cleaned


def clean_person_name(value: str) -> str:
    return _clean_name(value, kind="Name", extra=_PERSON_NAME_EXTRA, max_len=100)


def clean_org_name(value: str) -> str:
    return _clean_name(
        value, kind="Hospital name", extra=_ORG_NAME_EXTRA, max_len=120
    )


def clean_pincode(value: str) -> str:
    cleaned = re.sub(r"\s+", "", value or "")
    if not re.fullmatch(r"\d{6}", cleaned):
        raise ValueError("Pincode must be exactly 6 digits")
    return cleaned


def validate_password(value: str) -> str:
    if value is None or len(value) < MIN_PASSWORD_LENGTH or not value.strip():
        raise ValueError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
        )
    return value


def clean_org_name_optional(value: Optional[str]) -> Optional[str]:
    return None if value is None else clean_org_name(value)


def clean_pincode_optional(value: Optional[str]) -> Optional[str]:
    return None if value is None else clean_pincode(value)


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
class RegisterHospitalRequest(BaseModel):
    hospital_name: str
    address: Optional[str] = None
    pincode: str
    city: Optional[str] = None
    admin_name: str
    admin_email: str
    password: str

    _v_name = field_validator("hospital_name")(clean_org_name)
    _v_admin = field_validator("admin_name")(clean_person_name)
    _v_pin = field_validator("pincode")(clean_pincode)
    _v_pw = field_validator("password")(validate_password)


class LoginRequest(BaseModel):
    email: str
    password: str


class StaffCreate(BaseModel):
    """Admin-provisioned staff account (POST /users)."""

    name: str
    email: str
    role: StaffRole  # hospital_admin is created only via register-hospital
    temporary_password: str

    _v_name = field_validator("name")(clean_person_name)
    _v_pw = field_validator("temporary_password")(validate_password)


class StaffOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    email: str
    role: str
    must_change_password: bool
    created_at: datetime


class ResetPasswordRequest(BaseModel):
    new_password: str

    _v_pw = field_validator("new_password")(validate_password)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    _v_pw = field_validator("new_password")(validate_password)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    hospital_id: uuid.UUID
    name: str
    email: str
    role: str
    must_change_password: bool = False


class TokenResponse(BaseModel):
    token: str
    user: UserOut


class HospitalOut(BaseModel):
    """Public — GET /hospitals (kept for an "is my hospital already registered?"
    check; sign-up no longer uses a picker)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    city: Optional[str] = None
    pincode: Optional[str] = None


class HospitalDetailOut(BaseModel):
    """The caller's own hospital (GET /hospital) — powers the nav bar + profile."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    address: Optional[str] = None
    city: Optional[str] = None
    pincode: Optional[str] = None
    logo_url: Optional[str] = None
    created_at: datetime


class HospitalUpdate(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    pincode: Optional[str] = None

    _v_name = field_validator("name")(clean_org_name_optional)
    _v_pin = field_validator("pincode")(clean_pincode_optional)


# --------------------------------------------------------------------------- #
# Patients
# --------------------------------------------------------------------------- #
class PatientCreate(BaseModel):
    name: str
    dob: Optional[date] = None
    sex: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    consent_given_by: ConsentGivenBy
    consent_relationship: Optional[str] = None

    _v_name = field_validator("name")(clean_person_name)

    @model_validator(mode="after")
    def _require_relationship_for_guardian(self) -> "PatientCreate":
        """Rule 7: a guardian consenting has to say who they are to the patient.

        The register-patient form only shows the relationship field when
        "Guardian / carer" is picked, but that is a UI convenience — a consent
        record that says "a guardian consented" without naming the relationship
        isn't a usable consent record, so the rule is enforced here too.
        """
        if self.consent_given_by == "guardian" and not (self.consent_relationship or "").strip():
            raise ValueError(
                "consent_relationship is required when consent is given by a guardian"
            )
        return self


class PatientListItem(BaseModel):
    """One deduped patient row, with derived latest-visit context.

    ``derived_status``/``latest_*`` come from the most recent visit (§3: no stored
    patient-status column). ``latest_mri_status``/``latest_speech_status`` are only
    meaningful for the receptionist's incomplete-visits queue (what's missing).
    """

    id: uuid.UUID
    name: str
    dob: Optional[date] = None
    sex: Optional[str] = None
    phone: Optional[str] = None
    latest_visit_id: Optional[uuid.UUID] = None
    latest_visit_date: Optional[datetime] = None
    latest_visit_type: Optional[str] = None
    latest_visit_status: Optional[str] = None
    latest_doctor_diagnosis: Optional[str] = None
    latest_mri_status: Optional[str] = None
    latest_speech_status: Optional[str] = None


class TrendPoint(BaseModel):
    visit_date: datetime
    mmse: Optional[float] = None
    cdr: Optional[float] = None


class VisitSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    visit_type: str
    visit_date: datetime
    status: str
    mri_status: str
    speech_status: str
    model_prediction: Optional[str] = None
    doctor_diagnosis: Optional[str] = None


class PatientProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    hospital_id: uuid.UUID
    name: str
    dob: Optional[date] = None
    sex: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    consent_given_by: str
    consent_relationship: Optional[str] = None
    created_at: datetime
    visits: list[VisitSummary] = []
    trend: list[TrendPoint] = []


class NextVisitTypeOut(BaseModel):
    visit_type: VisitType
    reason: str


# --------------------------------------------------------------------------- #
# Visits
# --------------------------------------------------------------------------- #
class VisitCreate(BaseModel):
    patient_id: uuid.UUID
    visit_type: VisitType
    mmse: Optional[float] = None
    cdr: Optional[float] = None
    edu: Optional[float] = None  # screening only
    ses: Optional[float] = None  # screening only


class DiagnosisHistoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    doctor_diagnosis: str
    doctor_notes: Optional[str] = None
    saved_by_user_id: uuid.UUID
    saved_at: datetime


class VisitDetailOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    hospital_id: uuid.UUID
    visit_type: str
    visit_date: datetime
    created_by_user_id: uuid.UUID
    mmse: Optional[float] = None
    cdr: Optional[float] = None
    edu: Optional[float] = None
    ses: Optional[float] = None
    mri_status: str
    speech_status: str
    mri_feature_vector: Optional[list] = None
    speech_feature_vector: Optional[list] = None
    model_prediction: Optional[str] = None  # QSVM ("Quantum SVM")
    model_confidence: Optional[float] = None
    svm_prediction: Optional[str] = None  # classical SVM, display-only comparison
    svm_confidence: Optional[float] = None
    requires_review: bool
    status: str
    doctor_diagnosis: Optional[str] = None
    doctor_notes: Optional[str] = None
    diagnosis_saved_at: Optional[datetime] = None
    diagnosis_saved_by_user_id: Optional[uuid.UUID] = None
    agreement_flag: Optional[str] = None
    created_at: datetime
    diagnosis_history: list[DiagnosisHistoryOut] = []


class DiagnosisCreate(BaseModel):
    doctor_diagnosis: DoctorDiagnosis
    doctor_notes: Optional[str] = None
