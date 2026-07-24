"""Server-side visit-type decision (spec §4).

One function is the source of truth for "screening vs follow-up", used by both
``POST /visits`` validation and ``GET /patients/{id}/next-visit-type``. It is a
pure function of the patient's most recent *screening* visit — the caller (the
router) is responsible for fetching that via a Rule-12-scoped query and passing
it in, so this stays DB-agnostic and unit-testable.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.models import Visit

RESCREEN_INTERVAL_MONTHS = 12  # hardcoded constant for Phase 2 (§4)


def months_between(now: datetime, past: datetime) -> int:
    """Whole calendar months elapsed from ``past`` to ``now`` (>=0)."""
    months = (now.year - past.year) * 12 + (now.month - past.month)
    if now.day < past.day:
        months -= 1
    return max(months, 0)


def decide_visit_type(
    last_screening: Visit | None,
    *,
    force_screening: bool = False,
    now: datetime | None = None,
) -> tuple[str, str]:
    """Return ``(visit_type, reason)`` per §4.

    ``last_screening`` is the patient's most recent ``visit_type='screening'``
    visit, or None if they've never been screened.
    """
    now = now or datetime.now(timezone.utc)

    if force_screening:
        return "screening", "Forced full screening (manual override)."

    if last_screening is None:
        return "screening", "No prior screening on record."

    if last_screening.doctor_diagnosis is None:
        return "screening", "Last screening is still pending doctor review."

    if last_screening.doctor_diagnosis == "Needs further evaluation":
        return "screening", "Last diagnosis was inconclusive (not a confirmed status)."

    months = months_between(now, _as_utc(last_screening.visit_date))
    if months >= RESCREEN_INTERVAL_MONTHS:
        return (
            "screening",
            f"Due for re-screening ({months} months since last screening).",
        )

    return "follow_up", "Confirmed diagnosis on record and within the re-screen window."


def _as_utc(dt: datetime) -> datetime:
    """Coerce naive timestamps to UTC so month math is timezone-consistent."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
