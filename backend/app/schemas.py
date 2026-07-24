"""Pydantic request/response models for the §5 API contract.

Response models use ``from_attributes=True`` so they can be built straight from
SQLAlchemy ORM objects. Email fields are plain ``str`` (not ``EmailStr``) to
avoid pulling in ``email-validator`` — format validation isn't required for the
Phase 2 demo and the DB's UNIQUE(email) is the real guard.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

# Literals mirror the Postgres ENUM value sets in models.py / 001_init.sql.
StaffRole = Literal["receptionist", "clinician"]
ConsentGivenBy = Literal["patient", "guardian"]
VisitType = Literal["screening", "follow_up"]
DoctorDiagnosis = Literal["Nondemented", "Demented", "Needs further evaluation"]


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
class RegisterHospitalRequest(BaseModel):
    hospital_name: str
    address: Optional[str] = None
    admin_name: str
    admin_email: str
    password: str


class RegisterStaffRequest(BaseModel):
    hospital_id: uuid.UUID
    name: str
    email: str
    password: str
    role: StaffRole  # hospital_admin is created only via register-hospital


class LoginRequest(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    hospital_id: uuid.UUID
    name: str
    email: str
    role: str


class TokenResponse(BaseModel):
    token: str
    user: UserOut


class HospitalOut(BaseModel):
    """Public — powers the sign-up hospital picker (GET /hospitals)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str


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
