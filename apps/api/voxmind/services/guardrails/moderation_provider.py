"""Two real safety-classification implementations, matching the LLM/TTS
provider pattern: a real cloud API (credential-gated) and a real, honestly-
labeled-as-weaker deterministic fallback - never a fake "always safe"
placeholder. Unlike `build_llm_provider()`/`build_tts_provider()` (which
return `None` when unavailable), `build_moderation_provider()` always
returns something real: skipping safety scanning entirely when no API key
is configured would be a worse default than running the weaker keyword
check.
"""
from __future__ import annotations

import asyncio
import re

import structlog
from openai import OpenAI

from voxmind.core.config import Settings
from voxmind.services.guardrails.interfaces import ModerationProvider, SafetyScanResult

logger = structlog.get_logger(__name__)

# Deliberately high-level category triggers, not an exhaustive or
# instructive list - this fallback exists only to catch blatant cases when
# the real moderation API isn't configured, documented honestly as weaker
# than a real classifier (see docs/guardrails.md).
_KEYWORD_CATEGORIES: dict[str, re.Pattern[str]] = {
    "self_harm": re.compile(r"\bhow (to|do i) (kill myself|end my life|commit suicide)\b", re.IGNORECASE),
    "violence": re.compile(r"\bhow (to|do i) (make|build) a (bomb|explosive|weapon to kill)\b", re.IGNORECASE),
    "illegal_activity": re.compile(r"\bhow (to|do i) (synthesize|manufacture) (meth|heroin|fentanyl)\b", re.IGNORECASE),
}


class KeywordSafetyScanner:
    provider_name = "keyword_fallback"

    async def moderate(self, text: str) -> SafetyScanResult:
        categories = [name for name, pattern in _KEYWORD_CATEGORIES.items() if pattern.search(text)]
        return SafetyScanResult(is_unsafe=bool(categories), categories=categories, method="keyword_fallback")


class OpenAiModerationProvider:
    provider_name = "openai_moderation"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._fallback = KeywordSafetyScanner()

    async def moderate(self, text: str) -> SafetyScanResult:
        try:
            return await asyncio.to_thread(self._run, text)
        except Exception as exc:  # noqa: BLE001 - a moderation-call failure degrades to the
            # real keyword fallback rather than either blocking everything or approving
            # everything by default.
            # Never log str(exc) here: proven live that this SDK's own
            # AuthenticationError message includes a masked-but-partial echo
            # of the real API key. Only the exception type is safe to log.
            logger.warning("openai_moderation_failed_falling_back", error_type=type(exc).__name__)
            return await self._fallback.moderate(text)

    def _run(self, text: str) -> SafetyScanResult:
        response = self._client.moderations.create(model="omni-moderation-latest", input=text)
        result = response.results[0]
        flagged_categories = [
            category for category, flagged in result.categories.model_dump().items() if flagged
        ]
        return SafetyScanResult(
            is_unsafe=result.flagged, categories=flagged_categories, method="openai_moderation"
        )


def build_moderation_provider(settings: Settings) -> ModerationProvider:
    if settings.OPENAI_API_KEY:
        return OpenAiModerationProvider(settings)
    return KeywordSafetyScanner()
