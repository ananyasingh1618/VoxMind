"""Real TTS provider tests. `facebook/mms-tts-eng` is not gated - genuinely
downloads and runs, no credentials needed (marked `real_model`, same bar as
Whisper/Wav2Vec2/MiniLM). The OpenAI TTS test follows the `requires_hf_token`-
style pattern for a real, credential-gated provider - skipped without a
real OpenAI credential, never faked.

`OPENAI_API_KEY` alone is not sufficient to know that: this project also
supports pointing `OpenAiLlmProvider` at any OpenAI-Chat-Completions-
compatible endpoint via `OPENAI_BASE_URL` (e.g. Groq - see
docs/DECISIONS/0012), which reuses the same settings field for a
key that is real and working for chat completions, but is not a real
OpenAI credential and has no OpenAI TTS access at all. `OpenAiTtsProvider`
itself has no `base_url` override (TTS was never part of the Groq-reuse
change), so it always calls the real `api.openai.com`. The skip condition
below therefore requires `OPENAI_API_KEY` AND an unset `OPENAI_BASE_URL` -
i.e. a key that is actually still pointed at OpenAI itself - rather than
treating any `OPENAI_API_KEY` presence as proof of real OpenAI access.
"""
from __future__ import annotations

import io

import pytest
import soundfile as sf

from voxmind.core.config import get_settings
from voxmind.services.tts.local_hf_provider import LocalHfTtsProvider


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_local_tts_produces_real_non_silent_playable_audio():
    provider = LocalHfTtsProvider(get_settings())
    result = await provider.synthesize("VoxMind is a multimodal conversational intelligence platform.")

    assert result.content_type == "audio/wav"
    assert result.provider == "local_hf:facebook/mms-tts-eng"
    assert result.duration_ms > 0
    assert len(result.audio_bytes) > 1000

    samples, sample_rate = sf.read(io.BytesIO(result.audio_bytes))
    assert sample_rate > 0
    assert len(samples) > 0
    assert abs(samples).max() > 0.01  # genuinely non-silent, not an empty/fake clip


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_local_tts_rejects_empty_text():
    from voxmind.core.exceptions import PipelineProcessingError

    provider = LocalHfTtsProvider(get_settings())
    with pytest.raises(PipelineProcessingError):
        await provider.synthesize("   ")


def _has_real_openai_credential() -> bool:
    settings = get_settings()
    # A configured OPENAI_BASE_URL means OPENAI_API_KEY is deliberately
    # pointed at a different, OpenAI-compatible provider (e.g. Groq) - real
    # and working for chat completions, but not a real OpenAI credential,
    # and OpenAiTtsProvider has no base_url override to redirect it either.
    return bool(settings.OPENAI_API_KEY) and not settings.OPENAI_BASE_URL


@pytest.mark.requires_llm_credentials
@pytest.mark.asyncio
@pytest.mark.skipif(
    not _has_real_openai_credential(),
    reason="A real OpenAI credential (OPENAI_API_KEY with no OPENAI_BASE_URL override) is not "
    "configured in this environment - VoxMind's real, intentionally-configured TTS path here is "
    "the local facebook/mms-tts-eng provider instead (see test_local_tts_produces_real_non_silent_"
    "playable_audio above). Configure a real OpenAI key with no OPENAI_BASE_URL override and "
    "re-run with: pytest -m requires_llm_credentials",
)
async def test_real_openai_tts_produces_real_audio():
    from voxmind.services.tts.openai_provider import OpenAiTtsProvider

    provider = OpenAiTtsProvider(get_settings())
    result = await provider.synthesize("Hello from VoxMind.")

    assert result.content_type == "audio/mpeg"
    assert len(result.audio_bytes) > 1000
