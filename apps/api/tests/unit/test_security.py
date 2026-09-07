from __future__ import annotations

import base64
import uuid

import pytest
from jose import ExpiredSignatureError

from voxmind.core.config import get_settings
from voxmind.core.security import (
    create_access_token,
    decode_access_token,
    generate_csrf_token,
    generate_opaque_token,
    hash_password,
    hash_token,
    verify_password,
)


def test_password_hash_is_not_plaintext():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert hashed.startswith("$2b$")


def test_password_verification_round_trip():
    hashed = hash_password("s3cret-password")
    assert verify_password("s3cret-password", hashed) is True
    assert verify_password("wrong-password", hashed) is False


def test_access_token_round_trip():
    settings = get_settings()
    user_id = uuid.uuid4()
    token = create_access_token(user_id=user_id, settings=settings)
    payload = decode_access_token(token, settings=settings)
    assert payload["sub"] == str(user_id)
    assert payload["type"] == "access"


def _flip_signature_byte(token: str) -> str:
    """Tamper with a JWT by flipping every bit of the *decoded* first byte
    of the signature, rather than mutating a base64url character directly.

    Flipping the raw text of the last base64url character is not reliable:
    base64 groups can leave that character's low bit(s) unused/redundant
    (a function of how many bytes remain in the final group), so some
    substitutions decode back to the *same* signature bytes - which made
    this test intermittently pass or fail depending on the last character
    of a freshly generated token. Decoding first and XOR-ing a whole byte
    with 0xFF guarantees a different byte (`x ^ 0xFF` can never equal `x`
    for any 8-bit value), so the re-encoded signature is deterministically
    different every single run, regardless of the original token's bytes.
    """
    header_b64, payload_b64, signature_b64 = token.split(".")
    padded_signature = signature_b64 + "=" * (-len(signature_b64) % 4)
    raw_signature = bytearray(base64.urlsafe_b64decode(padded_signature))
    raw_signature[0] ^= 0xFF
    tampered_signature = base64.urlsafe_b64encode(bytes(raw_signature)).rstrip(b"=").decode("ascii")
    return f"{header_b64}.{payload_b64}.{tampered_signature}"


def test_access_token_rejects_tampering():
    settings = get_settings()
    token = create_access_token(user_id=uuid.uuid4(), settings=settings)
    tampered = _flip_signature_byte(token)
    assert tampered != token  # sanity: the tampering must actually change the token
    with pytest.raises(Exception):
        decode_access_token(tampered, settings=settings)


def test_expired_access_token_raises_expired_signature_error(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "ACCESS_TOKEN_TTL_MINUTES", -1)
    token = create_access_token(user_id=uuid.uuid4(), settings=settings)
    with pytest.raises(ExpiredSignatureError):
        decode_access_token(token, settings=settings)


def test_opaque_token_and_hash_are_deterministic_and_unique():
    token_a = generate_opaque_token()
    token_b = generate_opaque_token()
    assert token_a != token_b
    assert hash_token(token_a) == hash_token(token_a)
    assert hash_token(token_a) != hash_token(token_b)
    # the hash must never equal the raw token (i.e. we aren't accidentally storing plaintext)
    assert hash_token(token_a) != token_a


def test_csrf_tokens_are_unique():
    assert generate_csrf_token() != generate_csrf_token()
