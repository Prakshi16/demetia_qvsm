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
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.emails import clean_email_domain

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


def clean_optional_text(value: Optional[str], *, label: str, max_len: int) -> Optional[str]:
    """Trim free text; blank -> None; reject control characters and overlong input."""
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned == "":
        return None
    if len(cleaned) > max_len:
        raise ValueError(f"{label} must be at most {max_len} characters")
    if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in cleaned):
        raise ValueError(f"{label} contains control characters")
    return cleaned


# Phone: digits with an optional leading +, and the usual separators. 7–15
# digits is the E.164 range; separators are stripped before counting.
_PHONE_ALLOWED = re.compile(r"^\+?[\d\s\-().]+$")


def clean_phone(value: Optional[str]) -> Optional[str]:
    if value is None or value.strip() == "":
        return None
    cleaned = value.strip()
    if not _PHONE_ALLOWED.fullmatch(cleaned):
        raise ValueError("Phone may contain only digits, spaces, +, -, ( and )")
    digits = re.sub(r"\D", "", cleaned)
    if not 7 <= len(digits) <= 15:
        raise ValueError("Phone must have between 7 and 15 digits")
    return cleaned


_CITY_RE = re.compile(r"[^\W\d_][\w\s.'\-]*")


def clean_city(value: Optional[str]) -> Optional[str]:
    cleaned = clean_optional_text(value, label="City", max_len=80)
    if cleaned is not None and (not _CITY_RE.fullmatch(cleaned) or any(ch.isdigit() for ch in cleaned)):
        raise ValueError("City must contain only letters, spaces, . ' and -")
    return cleaned


def clean_address(value: Optional[str]) -> Optional[str]:
    return clean_optional_text(value, label="Address", max_len=300)


SEX_VALUES = ("Female", "Male", "Other", "Prefer not to say")

# Clinical scales. CDR is a fixed ordinal scale, not a continuous number.
CDR_VALUES = (0.0, 0.5, 1.0, 2.0, 3.0)
MMSE_RANGE = (0, 30)
EDU_RANGE = (0, 30)
SES_RANGE = (1, 5)

# strict=True rejects strings like "12" and booleans; allow_inf_nan=False
# rejects NaN/Infinity, which Python's JSON parser accepts as bare literals.
ClinicalNumber = Optional[Annotated[float, Field(strict=True, allow_inf_nan=False)]]


def _check_whole_in_range(value: Optional[float], label: str, low: int, high: int) -> Optional[float]:
    if value is None:
        return None
    if float(value) != int(value):
        raise ValueError(f"{label} must be a whole number")
    if not low <= value <= high:
        raise ValueError(f"{label} must be between {low} and {high}")
    return float(value)


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
    # The domain every account at this hospital will use. The admin's own login
    # address is derived from admin_name + this domain (never typed) — see
    # services/emails.py — so there is no admin_email field.
    email_domain: str
    password: str

    _v_name = field_validator("hospital_name")(clean_org_name)
    _v_admin = field_validator("admin_name")(clean_person_name)
    _v_pin = field_validator("pincode")(clean_pincode)
    _v_domain = field_validator("email_domain")(clean_email_domain)
    _v_pw = field_validator("password")(validate_password)
    _v_address = field_validator("address")(clean_address)
    _v_city = field_validator("city")(clean_city)


class LoginRequest(BaseModel):
    email: str
    password: str


# --------------------------------------------------------------------------- #
# Staff management (hospital_admin only) — §5. The admin never touches patient
# data; these endpoints are the admin's whole job: the clinicians and
# receptionists at their own hospital. Accounts are provisioned with a temporary
# password and land with must_change_password set (fault #5).
# --------------------------------------------------------------------------- #
class StaffCreate(BaseModel):
    name: str
    temporary_password: str
    role: StaffRole  # receptionist | clinician — an admin cannot mint another admin

    _v_name = field_validator("name")(clean_person_name)
    _v_pw = field_validator("temporary_password")(validate_password)
    # No email field: the login address is derived from `name` + the hospital's
    # email_domain (services/emails.allocate_email), so the whole roster is
    # consistent and the admin has nothing to mistype.


class StaffListItem(BaseModel):
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
    email_domain: Optional[str] = None  # immutable; shown, never edited
    created_at: datetime


class HospitalUpdate(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    pincode: Optional[str] = None

    _v_name = field_validator("name")(clean_org_name_optional)
    _v_pin = field_validator("pincode")(clean_pincode_optional)
    _v_address = field_validator("address")(clean_address)
    _v_city = field_validator("city")(clean_city)


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
    _v_phone = field_validator("phone")(clean_phone)
    _v_address = field_validator("address")(clean_address)

    @field_validator("dob")
    @classmethod
    def _v_dob(cls, value: Optional[date]) -> Optional[date]:
        if value is None:
            return None
        today = date.today()
        if value > today:
            raise ValueError("Date of birth cannot be in the future")
        if value.year < today.year - 120:
            raise ValueError("Date of birth is more than 120 years ago")
        return value

    @field_validator("sex")
    @classmethod
    def _v_sex(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value.strip() == "":
            return None
        if value not in SEX_VALUES:
            raise ValueError(f"Sex must be one of: {', '.join(SEX_VALUES)}")
        return value

    @field_validator("consent_relationship")
    @classmethod
    def _v_relationship(cls, value: Optional[str]) -> Optional[str]:
        cleaned = clean_optional_text(value, label="Relationship", max_len=60)
        if cleaned is not None and not all(ch.isalpha() or ch in " -'/" for ch in cleaned):
            raise ValueError("Relationship must be words only, e.g. Daughter")
        return cleaned

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
    """Clinical measures for a new visit.

    Every number is validated here, not only in the browser: whole-number MMSE
    0–30, CDR on its fixed scale, whole years of education 0–30, SES 1–5.
    Strings, booleans, NaN and Infinity are rejected outright. MMSE and CDR are
    required on both visit types; EDU and SES are required on a screening (the
    model needs them) and refused on a follow-up (it doesn't collect them), so
    nothing is silently dropped or imputed.
    """

    patient_id: uuid.UUID
    visit_type: VisitType
    mmse: ClinicalNumber = None
    cdr: ClinicalNumber = None
    edu: ClinicalNumber = None  # screening only
    ses: ClinicalNumber = None  # screening only

    @field_validator("mmse")
    @classmethod
    def _v_mmse(cls, value: Optional[float]) -> Optional[float]:
        return _check_whole_in_range(value, "MMSE", *MMSE_RANGE)

    @field_validator("cdr")
    @classmethod
    def _v_cdr(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if float(value) not in CDR_VALUES:
            raise ValueError("CDR must be one of 0, 0.5, 1, 2, 3")
        return float(value)

    @field_validator("edu")
    @classmethod
    def _v_edu(cls, value: Optional[float]) -> Optional[float]:
        return _check_whole_in_range(value, "Education (years)", *EDU_RANGE)

    @field_validator("ses")
    @classmethod
    def _v_ses(cls, value: Optional[float]) -> Optional[float]:
        return _check_whole_in_range(value, "Socioeconomic status", *SES_RANGE)

    @model_validator(mode="after")
    def _required_by_type(self) -> "VisitCreate":
        missing = [name for name in ("mmse", "cdr") if getattr(self, name) is None]
        if self.visit_type == "screening":
            missing += [name for name in ("edu", "ses") if getattr(self, name) is None]
        elif self.edu is not None or self.ses is not None:
            raise ValueError("edu and ses are collected on screening visits only")
        if missing:
            kind = self.visit_type.replace("_", "-")
            raise ValueError(f"Required for a {kind} visit: {', '.join(missing)}")
        return self


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
    model_prediction: Optional[str] = None  # classical SVM (2026-09-25 -- see prediction.py)
    model_confidence: Optional[float] = None
    # Legacy QSVM comparison columns. No longer written for new predictions
    # (see the 2026-09-25 note in prediction.py); kept for visits predicted
    # before that date and for schema/DB backward compatibility.
    svm_prediction: Optional[str] = None
    svm_confidence: Optional[float] = None
    # Research view (migration 006): the quantum-kernel SVM on the same row.
    qsvm_prediction: Optional[str] = None
    qsvm_confidence: Optional[float] = None
    mri_extraction: Optional[dict] = None
    requires_review: bool
    status: str
    doctor_diagnosis: Optional[str] = None
    doctor_notes: Optional[str] = None
    diagnosis_saved_at: Optional[datetime] = None
    diagnosis_saved_by_user_id: Optional[uuid.UUID] = None
    agreement_flag: Optional[str] = None
    created_at: datetime
    diagnosis_history: list[DiagnosisHistoryOut] = []


class VisitFileOut(BaseModel):
    """A short-lived signed URL for one of a visit's raw uploads.

    ``url`` points straight at Supabase Storage and expires after
    ``expires_in`` seconds — the client refetches on demand.
    """

    url: str
    filename: str
    content_type: str
    expires_in: int = 3600


class DiagnosisCreate(BaseModel):
    doctor_diagnosis: DoctorDiagnosis
    doctor_notes: Optional[str] = None

    @field_validator("doctor_notes")
    @classmethod
    def _v_notes(cls, value: Optional[str]) -> Optional[str]:
        return clean_optional_text(value, label="Notes", max_len=4000)


class ModelCard(BaseModel):
    """One classifier's cross-validated performance, for the comparison view."""

    key: Literal["svm", "qsvm"]
    name: str
    kernel: str
    canonical: bool
    metrics: dict[str, dict[str, float]]
    # |decision value| at the 95th percentile of the training set — the axis
    # extent for the decision-value chart, so the two models' very different
    # margin scales are each drawn against their own typical range.
    decision_scale: float


class ModelInfoOut(BaseModel):
    evaluation: str
    models: list[ModelCard]
