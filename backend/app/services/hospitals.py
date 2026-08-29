"""Branch-aware hospital-name uniqueness (fault #7).

Two branches of the same chain share a name but not a 6-digit pincode, so the
rule is: a given (lower(name), pincode) pair may exist only once. A matching name
at a *different* pincode is a legitimate new branch and is allowed. Mirrors the
``uq_hospitals_name_pincode`` index added in migration 003.
"""
from __future__ import annotations

import uuid

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import Hospital


def assert_hospital_name_available(
    db: Session,
    *,
    name: str,
    pincode: str | None,
    exclude_id: uuid.UUID | None = None,
) -> None:
    """Raise 409 if a hospital with this name already exists at this pincode."""
    query = db.query(Hospital).filter(func.lower(Hospital.name) == name.lower())
    if exclude_id is not None:
        query = query.filter(Hospital.id != exclude_id)

    for existing in query.all():
        if (existing.pincode or "") == (pincode or ""):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"A hospital named '{name}' at pincode {pincode} is already "
                    "registered. If you are a different branch, use that branch's "
                    "pincode; otherwise ask that hospital's admin for an account."
                ),
            )
