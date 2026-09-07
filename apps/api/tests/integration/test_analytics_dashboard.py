"""Real aggregate-query tests for `GET /analytics/dashboard` against a real
Postgres database - a fresh account must show genuine empty states (never
fabricated placeholder numbers), and real activity must show up as real
counts.
"""
from __future__ import annotations

import pytest

from voxmind.api.deps import get_llm_provider
from voxmind.core.config import get_settings
from voxmind.main import app
from voxmind.services.llm.mock_provider import MockLlmProvider

CREDENTIALS = {"email": "analytics-test@voxmind.dev", "password": "correct-horse-battery-staple"}


async def _authed_headers(client) -> dict:
    await client.post("/api/v1/auth/register", json=CREDENTIALS)
    login = await client.post("/api/v1/auth/login", json=CREDENTIALS)
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.fixture(autouse=True)
def _clear_llm_override():
    yield
    app.dependency_overrides.pop(get_llm_provider, None)


@pytest.mark.asyncio
async def test_fresh_account_shows_genuine_empty_states_not_fabricated_numbers(client):
    headers = await _authed_headers(client)

    response = await client.get("/api/v1/analytics/dashboard", headers=headers)
    assert response.status_code == 200
    body = response.json()

    assert body["conversations"]["total"] == 0
    assert body["latency"]["llm"]["sample_count"] == 0
    assert body["latency"]["llm"]["avg_ms"] is None
    assert body["emotion"]["by_label"] == []
    assert body["intelligence"]["sentiment_distribution"] == []
    assert body["intelligence"]["mismatch_average_score"] is None
    assert body["grounding"]["by_status"] == []
    assert body["failures"]["guardrail_blocked"] == 0


@pytest.mark.asyncio
async def test_real_activity_produces_real_dashboard_numbers(client):
    app.dependency_overrides[get_llm_provider] = lambda: MockLlmProvider()
    headers = await _authed_headers(client)

    conv = await client.post("/api/v1/conversations", json={"title": "analytics test"}, headers=headers)
    conversation_id = conv.json()["id"]
    ask = await client.post(
        f"/api/v1/conversations/{conversation_id}/ask", json={"question": "Hello VoxMind"}, headers=headers
    )
    assert ask.status_code == 201

    response = await client.get("/api/v1/analytics/dashboard", headers=headers)
    assert response.status_code == 200
    body = response.json()

    assert body["conversations"]["total"] == 1
    assert body["conversations"]["total_messages"] >= 2  # the user question + the assistant answer
    assert body["grounding"]["total_generations"] == 1
    assert body["latency"]["llm"]["sample_count"] == 1
    assert body["latency"]["llm"]["avg_ms"] is not None
    assert body["latency"]["retrieval"]["sample_count"] == 1


@pytest.mark.asyncio
async def test_pipeline_health_reflects_real_configuration(client):
    headers = await _authed_headers(client)
    response = await client.get("/api/v1/analytics/dashboard", headers=headers)
    health = response.json()["pipeline_health"]

    # LLM_PROVIDER is forced to "local_dev" for the whole test session
    # regardless of apps/api/.env (see conftest.py's os.environ.setdefault),
    # specifically so ordinary test runs never make a real, billed LLM call
    # through RagService - this remains genuinely true no matter what real
    # key is configured for the credential-gated real-model tests elsewhere.
    assert health["llm_provider"] == "local_dev"
    assert health["llm_provider_configured"] is False
    assert health["tts_provider"] == "local_hf"
    assert health["tts_provider_configured"] is True
    # moderation_provider is NOT forced by conftest.py - it reflects the
    # real `bool(settings.OPENAI_API_KEY)` check in
    # analytics_service.py:build_pipeline_health (see
    # services/guardrails/moderation_provider.py::build_moderation_provider,
    # the same real check). This environment now has a real OPENAI_API_KEY
    # configured (a genuine Groq key, for LLM generation - see
    # docs/DECISIONS/0012), so the honest, correct value is
    # "openai_moderation", even though that provider's own real moderation
    # call still gracefully falls back to the keyword scanner at call time
    # since Groq has no moderation endpoint. Asserting "keyword_fallback"
    # unconditionally would be the fabrication this project forbids, now
    # that a real key genuinely exists.
    expected_moderation_provider = "openai_moderation" if get_settings().OPENAI_API_KEY else "keyword_fallback"
    assert health["moderation_provider"] == expected_moderation_provider


@pytest.mark.asyncio
async def test_analytics_requires_authentication(client):
    response = await client.get("/api/v1/analytics/dashboard")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_analytics_only_reflects_the_calling_users_own_data(client):
    headers_a = await _authed_headers(client)
    await client.post("/api/v1/conversations", json={"title": "user a's conversation"}, headers=headers_a)

    client.cookies.clear()
    other_credentials = {"email": "analytics-intruder@voxmind.dev", "password": "another-password-123"}
    await client.post("/api/v1/auth/register", json=other_credentials)
    login_b = await client.post("/api/v1/auth/login", json=other_credentials)
    headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

    response = await client.get("/api/v1/analytics/dashboard", headers=headers_b)
    assert response.status_code == 200
    assert response.json()["conversations"]["total"] == 0
