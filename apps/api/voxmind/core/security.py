"""Pure cryptographic helpers: password hashing, JWT access tokens, and
refresh/CSRF token generation. No DB access happens in this module — rotation
and revocation logic (which needs a transaction) lives in
repositories/refresh_token_repository.py.
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from voxmind.core.config import Settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return _pwd_context.verify(plain_password, hashed_password)


def create_access_token(*, user_id: uuid.UUID, settings: Settings) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_TTL_MINUTES),
        "type": "access",
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str, *, settings: Settings) -> dict:
    """Raises jose.ExpiredSignatureError or jose.JWTError directly so callers
    can distinguish an expired token (→ session_expired) from a malformed/
    invalid one (→ invalid_credentials) instead of collapsing both into one
    generic error."""
    payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    if payload.get("type") != "access":
        raise JWTError("Token is not an access token.")
    return payload


def generate_opaque_token() -> str:
    """256 bits of randomness, URL-safe. Used for both refresh and WS tickets."""
    return secrets.token_urlsafe(32)


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)
