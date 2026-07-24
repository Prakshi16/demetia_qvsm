"""Patient endpoints (§5). Every read goes through ``get_scoped_query`` (Rule 12).

The two specialised queues (clinician pending-review, receptionist
incomplete-visits) and the general list all key off a patient's *most recent*
visit — §3 says derive status at query time, don't store a patient-status column.
``list_patients`` is the shared builder; the dashboard router reuses it.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import (
    CurrentUser,
    get_current_user,
    get_scoped_query,
    require_clinician,
    require_receptionist,
)
from app.models import Patient, Visit
from app.schemas import (
    NextVisitTypeOut,
    PatientCreate,
    PatientListItem,
    PatientProfileOut,
    TrendPoint,
    VisitSummary,
)
from app.services.audit import record_audit
from app.services.visit_logic import decide_visit_type

router = APIRouter(prefix="/patients", tags=["patients"])


# --------------------------------------------------------------------------- #
# Shared list-building helpers
# --------------------------------------------------------------------------- #
def _latest_visits_by_patient(
    db: Session, user: CurrentUser
) -> dict[uuid.UUID, Visit]:
    """Map each patient_id -> their most recent visit (scoped to the hospital)."""
    visits = (
        get_scoped_query(db, Visit, user).order_by(Visit.visit_date.desc()).all()
    )
    latest: dict[uuid.UUID, Visit] = {}
    for v in visits:
        latest.setdefault(v.patient_id, v)  # first seen == most recent
    return latest


def _matches(patient: Patient, search: str) -> bool:
    needle = search.lower().strip()
    fields = [patient.name or "", patient.phone or "", str(patient.id)]
    return any(needle in f.lower() for f in fields)


def _item(patient: Patient, latest: Visit | None) -> PatientListItem:
    return PatientListItem(
        id=patient.id,
        name=patient.name,
        dob=patient.dob,
        sex=patient.sex,
        phone=patient.phone,
        latest_visit_id=latest.id if latest else None,
        latest_visit_date=latest.visit_date if latest else None,
        latest_visit_type=latest.visit_type if latest else None,
        latest_visit_status=latest.status if latest else None,
        latest_doctor_diagnosis=latest.doctor_diagnosis if latest else None,
        latest_mri_status=latest.mri_status if latest else None,
        latest_speech_status=latest.speech_status if latest else None,
    )


def list_patients(
    db: Session,
    user: CurrentUser,
    search: str | None = None,
    keep: Callable[[Visit | None], bool] | None = None,
) -> list[PatientListItem]:
    """Deduped patient rows, most-recent-visit-desc.

    ``keep`` optionally filters on the patient's latest visit (used by the
    pending-review / incomplete-visits queues); patients with no visit are only
    included when ``keep`` is None (the general list).
    """
    patients = get_scoped_query(db, Patient, user).all()
    if search:
        patients = [p for p in patients if _matches(p, search)]

    latest_by_patient = _latest_visits_by_patient(db, user)
    items: list[PatientListItem] = []
    for p in patients:
        latest = latest_by_patient.get(p.id)
        if keep is not None and not keep(latest):
            continue
        items.append(_item(p, latest))

    # most-recent-visit first; patients with no visit sort last. The fallback is
    # tz-aware because DB timestamps are (comparing aware vs naive would raise).
    _min = datetime.min.replace(tzinfo=timezone.utc)
    items.sort(
        key=lambda i: (i.latest_visit_date is not None, i.latest_visit_date or _min),
        reverse=True,
    )
    return items


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("", response_model=list[PatientListItem])
def list_all_patients(
    search: str | None = Query(default=None),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> list[PatientListItem]:
    return list_patients(db, user, search)


@router.get("/pending-review", response_model=list[PatientListItem])
def pending_review(
    search: str | None = Query(default=None),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_clinician),
) -> list[PatientListItem]:
    """Clinician queue: latest visit is a complete screening awaiting a diagnosis."""

    def keep(latest: Visit | None) -> bool:
        return (
            latest is not None
            and latest.visit_type == "screening"
            and latest.status == "pending_review"
        )

    return list_patients(db, user, search, keep)


@router.get("/incomplete-visits", response_model=list[PatientListItem])
def incomplete_visits(
    search: str | None = Query(default=None),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_receptionist),
) -> list[PatientListItem]:
    """Receptionist queue: latest visit still missing a modality (resumable)."""

    def keep(latest: Visit | None) -> bool:
        return latest is not None and latest.status == "awaiting_uploads"

    return list_patients(db, user, search, keep)


@router.post("", response_model=PatientProfileOut, status_code=status.HTTP_201_CREATED)
def create_patient(
    body: PatientCreate,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> PatientProfileOut:
    """Register a patient (consent captured once here — Rule 7). No visit created."""
    patient = Patient(
        hospital_id=user.hospital_id,
        name=body.name,
        dob=body.dob,
        sex=body.sex,
        phone=body.phone,
        address=body.address,
        consent_given_by=body.consent_given_by,
        consent_relationship=body.consent_relationship,
    )
    db.add(patient)
    db.flush()
    record_audit(
        db,
        hospital_id=user.hospital_id,
        user_id=user.user_id,
        action="register_patient",
        target_type="patient",
        target_id=patient.id,
    )
    db.commit()
    db.refresh(patient)
    # Built explicitly (not from_attributes) — a fresh patient has no visits/trend.
    return PatientProfileOut(
        id=patient.id,
        hospital_id=patient.hospital_id,
        name=patient.name,
        dob=patient.dob,
        sex=patient.sex,
        phone=patient.phone,
        address=patient.address,
        consent_given_by=patient.consent_given_by,
        consent_relationship=patient.consent_relationship,
        created_at=patient.created_at,
        visits=[],
        trend=[],
    )


def _load_scoped_patient(db: Session, user: CurrentUser, patient_id: uuid.UUID) -> Patient:
    patient = (
        get_scoped_query(db, Patient, user).filter(Patient.id == patient_id).first()
    )
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
        )
    return patient


@router.get("/{patient_id}", response_model=PatientProfileOut)
def get_patient(
    patient_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> PatientProfileOut:
    """Profile + visit history (desc) + MMSE/CDR trend series (chronological)."""
    patient = _load_scoped_patient(db, user, patient_id)
    visits = (
        get_scoped_query(db, Visit, user)
        .filter(Visit.patient_id == patient_id)
        .order_by(Visit.visit_date.desc())
        .all()
    )
    trend = [
        TrendPoint(visit_date=v.visit_date, mmse=v.mmse, cdr=v.cdr)
        for v in sorted(visits, key=lambda v: v.visit_date)
        if v.mmse is not None or v.cdr is not None
    ]
    return PatientProfileOut(
        id=patient.id,
        hospital_id=patient.hospital_id,
        name=patient.name,
        dob=patient.dob,
        sex=patient.sex,
        phone=patient.phone,
        address=patient.address,
        consent_given_by=patient.consent_given_by,
        consent_relationship=patient.consent_relationship,
        created_at=patient.created_at,
        visits=[VisitSummary.model_validate(v) for v in visits],
        trend=trend,
    )


@router.get("/{patient_id}/next-visit-type", response_model=NextVisitTypeOut)
def next_visit_type(
    patient_id: uuid.UUID,
    force_screening: bool = Query(default=False),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> NextVisitTypeOut:
    """§4 decision: screening vs follow-up for this patient's next visit."""
    _load_scoped_patient(db, user, patient_id)
    last_screening = (
        get_scoped_query(db, Visit, user)
        .filter(Visit.patient_id == patient_id, Visit.visit_type == "screening")
        .order_by(Visit.visit_date.desc())
        .first()
    )
    visit_type, reason = decide_visit_type(
        last_screening, force_screening=force_screening, now=datetime.now(timezone.utc)
    )
    return NextVisitTypeOut(visit_type=visit_type, reason=reason)
