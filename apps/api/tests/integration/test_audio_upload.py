from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CREDENTIALS = {"email": "audio-upload@voxmind.dev", "password": "correct-horse-battery-staple"}


async def _authed_headers_and_conversation(client) -> tuple[dict, str]:
    await client.post("/api/v1/auth/register", json=CREDENTIALS)
    login = await client.post("/api/v1/auth/login", json=CREDENTIALS)
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    conversation = await client.post("/api/v1/conversations", json={"title": "audio test"}, headers=headers)
    return headers, conversation.json()["id"]


@pytest.mark.asyncio
async def test_upload_real_wav_creates_audio_asset(client):
    headers, conversation_id = await _authed_headers_and_conversation(client)
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()

    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/audio",
        files={"file": ("hello_world.wav", wav_bytes, "audio/wav")},
        headers=headers,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["original_filename"] == "hello_world.wav"
    assert body["content_type"] == "audio/wav"
    assert body["size_bytes"] == len(wav_bytes)
    # Duration/sample_rate are still null here - real preprocessing (which
    # measures them) only runs when /process is called, not at upload time.
    assert body["duration_ms"] is None


@pytest.mark.asyncio
async def test_upload_rejects_non_audio_content(client):
    headers, conversation_id = await _authed_headers_and_conversation(client)

    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/audio",
        files={"file": ("fake.wav", b"this is definitely not audio data", "audio/wav")},
        headers=headers,
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_audio_format"


@pytest.mark.asyncio
async def test_upload_ignores_client_supplied_content_type_and_sniffs_real_bytes(client):
    """The request declares audio/wav, but the actual bytes are an mp3 -
    upload must succeed and be classified by real content, not by the
    client's claim (which would have been wrong here)."""
    headers, conversation_id = await _authed_headers_and_conversation(client)
    mp3_bytes = (FIXTURES_DIR / "hello_world.mp3").read_bytes()

    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/audio",
        files={"file": ("hello_world.mp3", mp3_bytes, "audio/wav")},  # deliberately wrong claim
        headers=headers,
    )

    assert response.status_code == 201
    assert response.json()["content_type"] == "audio/mp3"


@pytest.mark.asyncio
async def test_upload_rejects_oversized_file(client, monkeypatch):
    from voxmind.core.config import get_settings

    monkeypatch.setattr(get_settings(), "MAX_AUDIO_UPLOAD_BYTES", 100)
    headers, conversation_id = await _authed_headers_and_conversation(client)
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    assert len(wav_bytes) > 100

    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/audio",
        files={"file": ("hello_world.wav", wav_bytes, "audio/wav")},
        headers=headers,
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "audio_too_large"


@pytest.mark.asyncio
async def test_upload_requires_authentication(client):
    response = await client.post(
        "/api/v1/conversations/00000000-0000-0000-0000-000000000000/audio",
        files={"file": ("x.wav", b"irrelevant", "audio/wav")},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_cannot_upload_to_another_users_conversation(client):
    headers_a, conversation_id = await _authed_headers_and_conversation(client)
    del headers_a

    client.cookies.clear()
    other_credentials = {"email": "intruder@voxmind.dev", "password": "another-password-123"}
    await client.post("/api/v1/auth/register", json=other_credentials)
    login_b = await client.post("/api/v1/auth/login", json=other_credentials)
    headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/audio",
        files={"file": ("hello_world.wav", wav_bytes, "audio/wav")},
        headers=headers_b,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_audio_returns_only_original_assets_for_the_conversation(client):
    headers, conversation_id = await _authed_headers_and_conversation(client)
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()

    await client.post(
        f"/api/v1/conversations/{conversation_id}/audio",
        files={"file": ("hello_world.wav", wav_bytes, "audio/wav")},
        headers=headers,
    )

    response = await client.get(f"/api/v1/conversations/{conversation_id}/audio", headers=headers)
    assert response.status_code == 200
    assert len(response.json()) == 1
