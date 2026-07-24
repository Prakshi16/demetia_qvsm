"""Request dependencies: current-user resolution, role guards, and the one
sanctioned way to read tenant-owned tables.

**Rule 12 (multi-tenant isolation).** Supabase RLS is intentionally off for the
Phase 2 timeline, so hospital isolation is enforced *here*, in the app layer.
``get_scoped_query`` is the ONLY sanctioned entrypoint for reading ``patients``/
``visits`` — it pre-filters by the caller's ``hospital_id``. A raw
``db.query(Patient)`` anywhere is a cross-hospital data leak, not just a bug.
``diagnosis_history`` has no ``hospital_id`` of its own; reach it only through its
parent visit's scoped query.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TypeVar

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Query, Session

from app.db import get_db
from app.models import Patient, Visit
from app.security import InvalidTokenError, decode_token

# auto_error=False so a *missing* Authorization header yields our own 401 (not
# HTTPBearer's default 403), keeping every auth failure a consistent 401.
_bearer = HTTPBearer(auto_error=False)

# The only models that carry hospital_id and are safe to scope directly.
ScopedModel = TypeVar("ScopedModel", Patient, Visit)


@dataclass(frozen=True)
class CurrentUser:
    """The authenticated caller, decoded from the JWT (§5 payload)."""

    user_id: uuid.UUID
    hospital_id: uuid.UUID
    role: str
    name: str


_UNAUTHENTICATED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> CurrentUser:
    """Resolve the caller from ``Authorization: Bearer <jwt>`` or 401."""
    if creds is None or not creds.credentials:
        raise _UNAUTHENTICATED
    try:
        claims = decode_token(creds.credentials)
        return CurrentUser(
            user_id=uuid.UUID(claims["user_id"]),
            hospital_id=uuid.UUID(claims["hospital_id"]),
            role=claims["role"],
            name=claims["name"],
        )
    except (InvalidTokenError, KeyError, ValueError):
        # bad signature, expired, or a malformed/incomplete payload
        raise _UNAUTHENTICATED


def _require_role(role: str):
    """Build a dependency that 403s unless the caller has ``role``."""

    def guard(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {role} role",
            )
        return user

    return guard


require_clinician = _require_role("clinician")
require_receptionist = _require_role("receptionist")


def get_scoped_query(
    db: Session, model: type[ScopedModel], user: CurrentUser
) -> Query:
    """Rule 12: a query over ``model`` pre-filtered to the caller's hospital.

    Use this for EVERY read of patients/visits. ``model`` must have a
    ``hospital_id`` column (Patient, Visit) — that's enforced by the type bound.
    """
    return db.query(model).filter(model.hospital_id == user.hospital_id)
