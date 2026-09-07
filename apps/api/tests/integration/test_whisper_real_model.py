"""Genuine end-to-end faster-whisper execution - downloads the real 'tiny'
model (from the non-gated Systran/faster-whisper-tiny HF repo, no token
required) on first run and transcribes real synthesized speech. This is
slow (model load dominates; ~1s of actual inference) and is explicitly
separated from the fast unit-test suite via the `real_model` marker.

Run explicitly with: pytest -m real_model
Excluded from a default `pytest` run via -m "not real_model" if desired;
included by default here since it requires no credentials and this
environment has verified network access to Hugging Face.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from voxmind.core.config import get_settings
from voxmind.services.speech.whisper_provider import FasterWhisperProvider

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_real_whisper_transcribes_genuine_speech():
    settings = get_settings()
    provider = FasterWhisperProvider(settings)

    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    # Strip the WAV header via soundfile inside the provider - pass raw wav
    # bytes as produced by preprocessing would produce them.
    result = await provider.transcribe(wav_bytes, sample_rate=16000)

    assert result.model_version == f"faster-whisper:{settings.WHISPER_MODEL_SIZE}"
    assert result.language == "en"
    assert result.language_probability is not None and result.language_probability > 0.5
    assert "quick brown fox" in result.text.lower()
    assert len(result.segments) >= 1
    assert result.segments[0].confidence is not None
