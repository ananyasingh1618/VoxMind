"""Builds the configured `TextToSpeechProvider`, or `None` when disabled/
unconfigured - mirrors `services/llm/factory.py` exactly. `None` means "no
audio for this turn", never a fabricated silent/placeholder clip.
"""
from __future__ import annotations

from voxmind.core.config import Settings
from voxmind.services.tts.interfaces import TextToSpeechProvider


def build_tts_provider(settings: Settings) -> TextToSpeechProvider | None:
    if settings.TTS_PROVIDER == "local_hf":
        from voxmind.services.tts.local_hf_provider import LocalHfTtsProvider

        return LocalHfTtsProvider(settings)

    if settings.TTS_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            return None
        from voxmind.services.tts.openai_provider import OpenAiTtsProvider

        return OpenAiTtsProvider(settings)

    return None  # "disabled"
