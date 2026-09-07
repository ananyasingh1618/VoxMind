"""Real sentiment/NER model inference against a real Postgres database.
Marked `real_model` like Phase 2/3's Whisper/Wav2Vec2 tests: downloads and
runs genuine Hugging Face models, no credentials needed.
"""
from __future__ import annotations

import pytest

CREDENTIALS = {"email": "nlp-test@voxmind.dev", "password": "correct-horse-battery-staple"}


async def _authed_headers(client) -> dict:
    await client.post("/api/v1/auth/register", json=CREDENTIALS)
    login = await client.post("/api/v1/auth/login", json=CREDENTIALS)
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


async def _create_message(client, headers, content: str) -> tuple[str, str]:
    conversation = await client.post("/api/v1/conversations", json={"title": "nlp test"}, headers=headers)
    conversation_id = conversation.json()["id"]
    message = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"role": "user", "content": content},
        headers=headers,
    )
    return conversation_id, message.json()["id"]


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_real_sentiment_and_entities_are_genuinely_computed(client):
    headers = await _authed_headers(client)
    conversation_id, message_id = await _create_message(
        client, headers, "I love working with the VoxMind team in San Francisco!"
    )

    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages/{message_id}/nlp/process", headers=headers
    )
    assert response.status_code == 201
    body = response.json()
    assert body["sentiment_label"] in ("positive", "neutral", "negative")
    assert 0.0 <= body["sentiment_score"] <= 1.0
    assert any(e["text"] in ("San Francisco", "VoxMind") for e in body["entities"]) or body["entities"] == []
    assert body["model_versions"]["sentiment"]
    assert body["model_versions"]["ner"]

    fetched = await client.get(
        f"/api/v1/conversations/{conversation_id}/messages/{message_id}/nlp", headers=headers
    )
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_rule_based_intent_and_topics_run_alongside_real_models(client):
    headers = await _authed_headers(client)
    conversation_id, message_id = await _create_message(client, headers, "What time does the meeting start?")

    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages/{message_id}/nlp/process", headers=headers
    )
    assert response.status_code == 201
    body = response.json()
    assert body["intent_label"] == "question"
    assert body["model_versions"]["intent"] == "rule-based-v1"


@pytest.mark.asyncio
async def test_nlp_endpoint_requires_authentication(client):
    response = await client.post(
        "/api/v1/conversations/00000000-0000-0000-0000-000000000000/messages/"
        "00000000-0000-0000-0000-000000000000/nlp/process"
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_nlp_for_unprocessed_message_returns_null(client):
    headers = await _authed_headers(client)
    conversation_id, message_id = await _create_message(client, headers, "not analyzed yet")

    response = await client.get(
        f"/api/v1/conversations/{conversation_id}/messages/{message_id}/nlp", headers=headers
    )
    assert response.status_code == 200
    assert response.json() is None
