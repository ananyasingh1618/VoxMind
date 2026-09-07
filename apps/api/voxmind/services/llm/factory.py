"""Builds the configured `LlmProvider`, or `None` when unavailable - "no
provider" is a real, expected state (no API key configured), never papered
over with a fake provider that returns fabricated answers. Callers
(`RagService`) must treat `None` as "generation unavailable" and persist
that honestly, exactly like Phase 2/3's "unavailable" pattern for
diarization/emotion.
"""
from __future__ import annotations

from voxmind.core.config import Settings
from voxmind.services.llm.interfaces import LlmProvider


def build_llm_provider(settings: Settings) -> LlmProvider | None:
    if settings.MOCK_LLM:
        from voxmind.services.llm.mock_provider import MockLlmProvider

        return MockLlmProvider()

    if settings.LLM_PROVIDER == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            return None
        from voxmind.services.llm.anthropic_provider import AnthropicLlmProvider

        return AnthropicLlmProvider(settings)

    if settings.LLM_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            return None
        from voxmind.services.llm.openai_provider import OpenAiLlmProvider

        return OpenAiLlmProvider(settings)

    # LLM_PROVIDER == "local_dev": a documented config seam only, no
    # implementation exists (no offline/local model has been wired in) -
    # see docs/architecture.md's LLM strategy section.
    return None
