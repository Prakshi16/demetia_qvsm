"""Hospital email domain + name-derived login addresses.

Every account at a hospital shares one email domain, chosen once by the admin at
``register-hospital`` and immutable afterwards (a company has one domain). The
local part is derived from the person's name — first two name tokens, letters
only, joined by a dot: "Diya Sharma" -> ``diya.sharma``, "Prakshi Rani
Lakhchaura" -> ``prakshi.rani``, "Madonna" -> ``madonna``.

Login addresses are never typed by a user (not the admin's own, not staff): the
admin supplies a name and the address falls out of it, so the whole hospital is
consistent and there is nothing to mistype. On a collision the address is
shortened progressively (``diya.sharma`` -> ``diya.s`` -> ``diya.sh`` -> …) and,
only if every prefix is taken, a numeric suffix is appended.

These addresses are identifiers, not mailboxes — the demo sends no email. The
admin hands new staff their credentials on a trusted channel (fault #5).
"""
from __future__ import annotations

import re

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import User

# A pragmatic domain check: one or more dot-separated labels then a 2+ letter TLD,
# total length <= 253. Not RFC-complete — it only has to reject typos like a bare
# "gmail" or a stray space, not adjudicate every valid domain on the internet.
_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$"
)
_NON_LETTER = re.compile(r"[^a-z]")


def clean_email_domain(value: str) -> str:
    """Normalise and validate the hospital's staff email domain.

    Tolerates a pasted ``@domain``, a full ``someone@domain`` address, or a URL —
    anyone filling this in is likely to reach for one of those.
    """
    cleaned = (value or "").strip().lower()
    if "://" in cleaned:
        cleaned = cleaned.split("://", 1)[1]
    cleaned = cleaned.split("/", 1)[0].lstrip("@")
    if "@" in cleaned:
        cleaned = cleaned.split("@", 1)[1]
    if not _DOMAIN_RE.match(cleaned):
        raise ValueError(
            "Enter a valid email domain for your staff, e.g. yourhospital.com"
        )
    return cleaned


def local_part_from_name(name: str) -> str:
    """First two name tokens, letters only, dot-joined. Raises if nothing is left."""
    tokens = [_NON_LETTER.sub("", token.lower()) for token in name.split()]
    tokens = [token for token in tokens if token][:2]
    if not tokens:
        raise ValueError("This name has no letters to build an email address from")
    return ".".join(tokens)


def _candidates(local: str):
    """Yield login-name candidates in preference order (see module docstring)."""
    yield local
    if "." in local:
        first, last = local.split(".", 1)
        for length in range(1, len(last)):  # diya.s, diya.sh, … diya.sharm
            yield f"{first}.{last[:length]}"
    for suffix in range(2, 1000):  # last resort: diya.sharma2, diya.sharma3, …
        yield f"{local}{suffix}"


def allocate_email(db: Session, *, name: str, domain: str) -> str:
    """Pick the first name-derived ``local@domain`` that no user already holds."""
    local = local_part_from_name(name)
    taken = {
        row[0]
        for row in db.query(func.lower(User.email))
        .filter(func.lower(User.email).like(f"%@{domain}"))
        .all()
    }
    for candidate in _candidates(local):
        email = f"{candidate}@{domain}"
        if email not in taken:
            return email
    # _candidates yields ~1000 options; exhausting them needs 1000 same-name users.
    raise ValueError(f"Could not allocate a login address for '{name}' @ {domain}")
