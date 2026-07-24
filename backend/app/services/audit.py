"""Minimal audit trail (§3): patient registration, diagnosis saves, hospital
data deletion. One helper so every audited action logs the same shape.
"""
from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.models import AuditLog


def record_audit(
    db: Session,
    *,
    hospital_id: uuid.UUID,
    user_id: uuid.UUID | None,
    action: str,
    target_type: str | None = None,
    target_id: uuid.UUID | str | None = None,
) -> None:
    """Add an audit row to the session (caller commits)."""
    db.add(
        AuditLog(
            hospital_id=hospital_id,
            user_id=user_id,
            action=action,
            target_type=target_type,
            target_id=str(target_id) if target_id is not None else None,
        )
    )
