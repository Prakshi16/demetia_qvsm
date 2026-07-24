"""Password hashing and JWT issue/verify.

Phase 2 auth is deliberately a single self-issued JWT with a 24h expiry and no
refresh token (§2 simplification note). The token payload is exactly what §5
specifies — ``{user_id, hospital_id, role, name}`` — plus the standard ``exp``.
Everything here reads its knobs from ``settings`` (JWT_SECRET / JWT_ALGORITHM /
JWT_EXPIRE_HOURS) so there are no magic constants.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings

# bcrypt via passlib. bcrypt is pinned to 4.2.1 in requirements.txt because
# passlib 1.7.4's version probe breaks on bcrypt>=4.3.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(
    *, user_id: uuid.UUID, hospital_id: uuid.UUID, role: str, name: str
) -> str:
    """Sign a JWT carrying the §5 payload. Expiry is JWT_EXPIRE_HOURS from now."""
    now = datetime.now(timezone.utc)
    payload = {
        "user_id": str(user_id),
        "hospital_id": str(hospital_id),
        "role": role,
        "name": name,
        "exp": now + timedelta(hours=settings.JWT_EXPIRE_HOURS),
        "iat": now,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Verify signature + expiry and return the claims. Raises JWTError if invalid."""
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])


# Re-exported so callers can `except InvalidTokenError` without importing jose.
InvalidTokenError = JWTError
