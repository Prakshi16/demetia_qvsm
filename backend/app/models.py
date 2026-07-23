"""SQLAlchemy models mirroring the data model in TO_build_phase_2.md §3.

These map 1:1 onto the tables created by migrations/001_init.sql. The SQL file is
the source of truth for the schema on Supabase; these classes are how the FastAPI
endpoints read/write it. Keep the two in sync when the schema changes.

Enum note: the columns use native Postgres ENUM types (created in the migration).
`create_type=False` here tells SQLAlchemy the type already exists so it never tries
to CREATE TYPE itself — the migration owns that.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# --- Enum value sets (must match the CREATE TYPE statements in 001_init.sql) ---
ROLE = ("receptionist", "clinician", "hospital_admin")
CONSENT_GIVEN_BY = ("patient", "guardian")
VISIT_TYPE = ("screening", "follow_up")
MRI_STATUS = ("not_applicable", "idle", "uploading", "processing", "done")
SPEECH_STATUS = ("not_applicable", "idle", "recording", "processing", "done")
MODEL_PREDICTION = ("Nondemented", "Demented")
VISIT_STATUS = ("awaiting_uploads", "pending_review", "reviewed", "completed")
DOCTOR_DIAGNOSIS = ("Nondemented", "Demented", "Needs further evaluation")
AGREEMENT_FLAG = ("match", "mismatch")


def _enum(*values: str, name: str) -> Enum:
    """Reference an existing Postgres ENUM type without re-creating it."""
    return Enum(*values, name=name, create_type=False)


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Hospital(Base):
    __tablename__ = "hospitals"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String, nullable=False)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    users: Mapped[list[User]] = relationship(back_populates="hospital")
    patients: Mapped[list[Patient]] = relationship(back_populates="hospital")


class User(Base):
    """Staff account. Role is chosen once at sign-up (Product Rule 8)."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("hospitals.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(_enum(*ROLE, name="role"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    hospital: Mapped[Hospital] = relationship(back_populates="users")


class Patient(Base):
    """Consent is captured once here, at registration (Product Rule 7)."""

    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = _uuid_pk()
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("hospitals.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    sex: Mapped[str | None] = mapped_column(String, nullable=True)
    phone: Mapped[str | None] = mapped_column(String, nullable=True)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    consent_given_by: Mapped[str] = mapped_column(
        _enum(*CONSENT_GIVEN_BY, name="consent_given_by"), nullable=False
    )
    consent_relationship: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    hospital: Mapped[Hospital] = relationship(back_populates="patients")
    visits: Mapped[list[Visit]] = relationship(back_populates="patient")


class Visit(Base):
    """A screening or follow-up visit. See §3 for the full lifecycle of `status`."""

    __tablename__ = "visits"

    id: Mapped[uuid.UUID] = _uuid_pk()
    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id"), nullable=False, index=True
    )
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("hospitals.id"), nullable=False, index=True
    )
    visit_type: Mapped[str] = mapped_column(
        _enum(*VISIT_TYPE, name="visit_type"), nullable=False
    )
    visit_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )

    # --- clinical (both visit types collect mmse + cdr; edu/ses are screening-only) ---
    mmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    cdr: Mapped[float | None] = mapped_column(Float, nullable=True)
    edu: Mapped[float | None] = mapped_column(Float, nullable=True)
    ses: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- modality state (screening only) ---
    mri_status: Mapped[str] = mapped_column(
        _enum(*MRI_STATUS, name="mri_status"), nullable=False, default="not_applicable"
    )
    speech_status: Mapped[str] = mapped_column(
        _enum(*SPEECH_STATUS, name="speech_status"),
        nullable=False,
        default="not_applicable",
    )
    mri_feature_vector: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    speech_feature_vector: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # --- model output (screening only, filled once all 3 modalities are done) ---
    model_prediction: Mapped[str | None] = mapped_column(
        _enum(*MODEL_PREDICTION, name="model_prediction"), nullable=True
    )
    model_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- doctor review (screening only) ---
    requires_review: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    status: Mapped[str] = mapped_column(
        _enum(*VISIT_STATUS, name="visit_status"), nullable=False
    )
    doctor_diagnosis: Mapped[str | None] = mapped_column(
        _enum(*DOCTOR_DIAGNOSIS, name="doctor_diagnosis"), nullable=True
    )
    doctor_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    diagnosis_saved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    diagnosis_saved_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    agreement_flag: Mapped[str | None] = mapped_column(
        _enum(*AGREEMENT_FLAG, name="agreement_flag"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    patient: Mapped[Patient] = relationship(back_populates="visits")
    diagnosis_history: Mapped[list[DiagnosisHistory]] = relationship(
        back_populates="visit", order_by="DiagnosisHistory.saved_at"
    )


class DiagnosisHistory(Base):
    """Append-only log: one row per diagnosis save (Product Rule 5, same-day revision).

    visits.doctor_* columns mirror the most recent row here; this table is the
    audit trail so a same-day revision never silently loses the prior diagnosis.
    """

    __tablename__ = "diagnosis_history"

    id: Mapped[uuid.UUID] = _uuid_pk()
    visit_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("visits.id"), nullable=False, index=True
    )
    doctor_diagnosis: Mapped[str] = mapped_column(
        _enum(*DOCTOR_DIAGNOSIS, name="doctor_diagnosis"), nullable=False
    )
    doctor_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    saved_by_user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )
    saved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    visit: Mapped[Visit] = relationship(back_populates="diagnosis_history")


class AuditLog(Base):
    """Minimal audit: patient registration, diagnosis saves, hospital data deletion."""

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = _uuid_pk()
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("hospitals.id"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    action: Mapped[str] = mapped_column(String, nullable=False)
    target_type: Mapped[str | None] = mapped_column(String, nullable=True)
    target_id: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
