"""Real Anthropic/OpenAI network calls - skipped unless a real API key is
configured, exactly the `requires_hf_token`-style pattern
test_diarization_real_model.py established for a gated dependency this
environment doesn't have. No credentials are configured here, so these are
expected to skip - implemented and ready to run for real the moment a key is
added, never faked to appear "passing" without one.
"""
from __future__ import annotations

import pytest

from voxmind.core.config import get_settings
from voxmind.services.llm.interfaces import ConversationContext


def _context() -> ConversationContext:
    return ConversationContext(
        transcript="In one sentence, what is 2 + 2?",
        speaker_info=[],
        emotion=[],
        nlp=None,
        incongruence=[],
        memory=None,
        retrieved_knowledge=[],
        system_instructions="Answer briefly and factually.",
        safety_context={},
    )


@pytest.mark.requires_llm_credentials
@pytest.mark.asyncio
@pytest.mark.skipif(
    not get_settings().ANTHROPIC_API_KEY,
    reason="ANTHROPIC_API_KEY is not configured in this environment. Set it and re-run with: "
    "pytest -m requires_llm_credentials",
)
async def test_real_anthropic_generation_returns_structured_response():
    from voxmind.services.llm.anthropic_provider import AnthropicLlmProvider

    provider = AnthropicLlmProvider(get_settings())
    response = await provider.generate(_context())

    assert response.answer.strip()
    assert 0.0 <= response.confidence <= 1.0
    assert isinstance(response.citations, list)


@pytest.mark.requires_llm_credentials
@pytest.mark.asyncio
@pytest.mark.skipif(
    not get_settings().OPENAI_API_KEY,
    reason="OPENAI_API_KEY is not configured in this environment. Set it and re-run with: "
    "pytest -m requires_llm_credentials",
)
async def test_real_openai_generation_returns_structured_response():
    from voxmind.services.llm.openai_provider import OpenAiLlmProvider

    provider = OpenAiLlmProvider(get_settings())
    response = await provider.generate(_context())

    assert response.answer.strip()
    assert 0.0 <= response.confidence <= 1.0
    assert isinstance(response.citations, list)
