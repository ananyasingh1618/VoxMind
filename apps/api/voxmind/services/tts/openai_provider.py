"""Real OpenAI text-to-speech (`audio.speech.create`). Requires
`OPENAI_API_KEY` (a developer-console key, not a ChatGPT Plus subscription -
see docs/rag.md's LLM section for the same distinction). Returns real MP3
bytes directly from the API - no local encoding needed.
"""
from __future__ import annotations

import asyncio

import structlog
from openai import OpenAI

from voxmind.core.config import Settings
from voxmind.core.exceptions import PipelineProcessingError
from voxmind.services.tts.interfaces import TtsResult

logger = structlog.get_logger(__name__)


class OpenAiTtsProvider:
    provider_name = "openai"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def synthesize(self, text: str, voice_profile: str | None = None) -> TtsResult:
        if not text.strip():
            raise PipelineProcessingError("Cannot synthesize empty text.")
        try:
            return await asyncio.to_thread(self._run, text, voice_profile)
        except PipelineProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("openai_tts_failed", error=str(exc))
            raise PipelineProcessingError("OpenAI TTS synthesis failed.") from exc

    def _run(self, text: str, voice_profile: str | None) -> TtsResult:
        response = self._client.audio.speech.create(
            model=self._settings.OPENAI_TTS_MODEL,
            voice=voice_profile or self._settings.OPENAI_TTS_VOICE,
            input=text,
        )
        audio_bytes = response.read()
        # OpenAI's TTS API doesn't return audio duration - a rough estimate
        # (average speaking rate, ~15 chars/second) is used only for display/
        # instrumentation, never for playback control (the browser reads the
        # real duration from the audio file itself).
        estimated_ms = round(len(text) / 15 * 1000)
        return TtsResult(
            audio_bytes=audio_bytes,
            content_type="audio/mpeg",
            provider=f"openai:{self._settings.OPENAI_TTS_MODEL}",
            duration_ms=estimated_ms,
        )
