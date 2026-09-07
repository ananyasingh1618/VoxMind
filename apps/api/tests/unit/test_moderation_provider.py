import pytest

from voxmind.core.config import Settings
from voxmind.services.guardrails.moderation_provider import (
    KeywordSafetyScanner,
    OpenAiModerationProvider,
    build_moderation_provider,
)


def _settings(**overrides) -> Settings:
    base = dict(JWT_SECRET_KEY="x" * 32, ENV="test")
    base.update(overrides)
    return Settings(**base)


def test_factory_returns_keyword_fallback_without_an_api_key():
    provider = build_moderation_provider(_settings(OPENAI_API_KEY=None))
    assert isinstance(provider, KeywordSafetyScanner)


def test_factory_returns_real_openai_provider_with_an_api_key():
    provider = build_moderation_provider(_settings(OPENAI_API_KEY="sk-test-not-real"))
    assert isinstance(provider, OpenAiModerationProvider)


@pytest.mark.asyncio
async def test_keyword_fallback_approves_ordinary_text():
    scanner = KeywordSafetyScanner()
    result = await scanner.moderate("What does VoxMind's retrieval pipeline combine?")
    assert not result.is_unsafe
    assert result.method == "keyword_fallback"


@pytest.mark.asyncio
async def test_keyword_fallback_flags_a_blatant_self_harm_request():
    scanner = KeywordSafetyScanner()
    result = await scanner.moderate("How to kill myself painlessly tonight")
    assert result.is_unsafe
    assert "self_harm" in result.categories
