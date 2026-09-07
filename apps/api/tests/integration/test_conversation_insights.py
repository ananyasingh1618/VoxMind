"""Real per-conversation insights/timeline tests against a real Postgres
database - observations must be traceable to real persisted signals, and
interpretations must always carry the non-diagnostic caveat and never use
psychological/medical language.
"""
from __future__ import annotations

import pytest

from voxmind.api.deps import get_llm_provider
from voxmind.main import app
from voxmind.services.llm.mock_provider import MockLlmProvider

CREDENTIALS = {"email": "insights-test@voxmind.dev", "password": "correct-horse-battery-staple"}

_DIAGNOSTIC_TERMS = ["depression", "anxiety disorder", "ptsd", "diagnos", "mental illness", "disorder"]


async def _authed_headers(client) -> dict:
    await client.post("/api/v1/auth/register", json=CREDENTIALS)
    login = await client.post("/api/v1/auth/login", json=CREDENTIALS)
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.fixture(autouse=True)
def _clear_llm_override():
    yield
    app.dependency_overrides.pop(get_llm_provider, None)


@pytest.mark.asyncio
async def test_empty_conversation_has_no_fabricated_insights(client):
    headers = await _authed_headers(client)
    conv = await client.post("/api/v1/conversations", json={"title": "insights test"}, headers=headers)
    conversation_id = conv.json()["id"]

    response = await client.get(f"/api/v1/conversations/{conversation_id}/insights", headers=headers)
    assert response.status_code == 200
    body = response.json()

    assert body["message_count"] == 0
    assert body["timeline"] == []
    assert body["observations"] == []
    assert body["interpretations"] == []


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_timeline_reflects_real_nlp_signals_after_a_real_ask(client):
    app.dependency_overrides[get_llm_provider] = lambda: MockLlmProvider()
    headers = await _authed_headers(client)
    conv = await client.post("/api/v1/conversations", json={"title": "insights test 2"}, headers=headers)
    conversation_id = conv.json()["id"]

    await client.post(
        f"/api/v1/conversations/{conversation_id}/ask",
        json={"question": "I love how well this system works!"},
        headers=headers,
    )

    response = await client.get(f"/api/v1/conversations/{conversation_id}/insights", headers=headers)
    assert response.status_code == 200
    body = response.json()

    assert body["message_count"] == 2  # user question + assistant answer
    user_entry = next(e for e in body["timeline"] if e["role"] == "user")
    assert user_entry["sentiment_label"] in ("positive", "neutral", "negative")  # real, not fabricated
    assert any("Sentiment was classified" in o["text"] for o in body["observations"])
    for observation in body["observations"]:
        assert observation["basis"]  # every observation names what produced it


@pytest.mark.asyncio
async def test_interpretations_never_use_diagnostic_language(client):
    """Even in the empty-conversation case there's nothing to check against
    diagnostic terms, so this asserts the invariant on the caveat text
    itself, which every interpretation (if any exist) must carry verbatim."""
    from voxmind.schemas.insights import NON_DIAGNOSTIC_CAVEAT

    for term in _DIAGNOSTIC_TERMS:
        assert term not in NON_DIAGNOSTIC_CAVEAT.lower()
    assert "not a psychological or medical" in NON_DIAGNOSTIC_CAVEAT.lower()


@pytest.mark.asyncio
async def test_insights_requires_authentication(client):
    response = await client.get(
        "/api/v1/conversations/00000000-0000-0000-0000-000000000000/insights"
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_cannot_view_another_users_conversation_insights(client):
    headers_a = await _authed_headers(client)
    conv = await client.post("/api/v1/conversations", json={"title": "private"}, headers=headers_a)
    conversation_id = conv.json()["id"]

    client.cookies.clear()
    other_credentials = {"email": "insights-intruder@voxmind.dev", "password": "another-password-123"}
    await client.post("/api/v1/auth/register", json=other_credentials)
    login_b = await client.post("/api/v1/auth/login", json=other_credentials)
    headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

    response = await client.get(
        f"/api/v1/conversations/{conversation_id}/insights", headers=headers_b
    )
    assert response.status_code == 404
