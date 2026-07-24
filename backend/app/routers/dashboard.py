"""Role-aware dashboard (§5). Returns the same deduped, most-recent-visit-desc
patient list for both roles — the specialised queues (pending-review,
incomplete-visits) are their own endpoints, not filters of this one (Rule 9).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import CurrentUser, get_current_user
from app.routers.patients import list_patients
from app.schemas import PatientListItem

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_model=list[PatientListItem])
def dashboard(
    search: str | None = Query(default=None),
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> list[PatientListItem]:
    return list_patients(db, user, search)
