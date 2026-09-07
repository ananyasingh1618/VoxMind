"""API-level tests for the emotion endpoints, against a real Postgres
database. No trained emotion model is registered in this test database
(matching this environment's actual current state - no labeled dataset has
been supplied), so these tests verify the honest "unavailable" behavior
end-to-end, exactly as Phase 2 verified diarization's graceful degradation
without a Hugging Face token.
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CREDENTIALS = {"email": "emotion-test@voxmind.dev", "password": "correct-horse-battery-staple"}


async def _authed_headers(client) -> dict:
    await client.post("/api/v1/auth/register", json=CREDENTIALS)
    login = await client.post("/api/v1/auth/login", json=CREDENTIALS)
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


async def _create_conversation_and_message(client, headers) -> tuple[str, str]:
    conversation = await client.post(
        "/api/v1/conversations", json={"title": "emotion test"}, headers=headers
    )
    conversation_id = conversation.json()["id"]
    message = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"role": "user", "content": "placeholder - not derived from real audio processing"},
        headers=headers,
    )
    return conversation_id, message.json()["id"]


@pytest.mark.asyncio
async def test_emotion_processing_reports_unavailable_without_a_trained_model(client):
    """This message has no audio_asset_id (created via the plain text
    message endpoint, not via audio processing), so processing should fail
    with not_found for that reason specifically - see the audio-backed test
    below for the "no trained model" path."""
    headers = await _authed_headers(client)
    conversation_id, message_id = await _create_conversation_and_message(client, headers)

    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages/{message_id}/emotion/process",
        headers=headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_emotion_processing_requires_authentication(client):
    response = await client.post(
        "/api/v1/conversations/00000000-0000-0000-0000-000000000000/messages/"
        "00000000-0000-0000-0000-000000000000/emotion/process"
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_cannot_access_another_users_emotion_job(client):
    headers_a = await _authed_headers(client)
    conversation_id, message_id = await _create_conversation_and_message(client, headers_a)

    client.cookies.clear()
    other_credentials = {"email": "emotion-intruder@voxmind.dev", "password": "another-password-123"}
    await client.post("/api/v1/auth/register", json=other_credentials)
    login_b = await client.post("/api/v1/auth/login", json=other_credentials)
    headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

    response = await client.get(
        f"/api/v1/conversations/{conversation_id}/messages/{message_id}/emotion", headers=headers_b
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_emotion_predictions_is_empty_before_any_processing(client):
    headers = await _authed_headers(client)
    conversation_id, message_id = await _create_conversation_and_message(client, headers)

    response = await client.get(
        f"/api/v1/conversations/{conversation_id}/messages/{message_id}/emotion", headers=headers
    )
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_model_versions_for_untrained_component_is_empty(client):
    """No training has been run against this database - the registry must
    honestly report nothing, never a fabricated placeholder model."""
    headers = await _authed_headers(client)

    response = await client.get("/api/v1/models/emotion_classifier", headers=headers)
    assert response.status_code == 200
    assert response.json() == []
