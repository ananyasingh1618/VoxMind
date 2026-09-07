"""Tests the diarization provider's configuration guard - this runs for
real, without any Hugging Face credentials or network access, because it
verifies the code path taken when HUGGINGFACE_TOKEN is absent, which is
exactly this environment's actual condition. The real-model path (an
authenticated pipeline actually running inference) is a separate, clearly
marked test - see test_diarization_real_model.py - skipped here since no
token is configured. See docs/audio.md for what running it for real
requires.
"""
from __future__ import annotations

import pytest

from voxmind.core.config import get_settings
from voxmind.core.exceptions import AudioProcessingError
from voxmind.services.speech.diarization_provider import PyannoteDiarizationProvider


@pytest.mark.asyncio
async def test_diarize_raises_clear_error_without_huggingface_token(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "HUGGINGFACE_TOKEN", None)
    provider = PyannoteDiarizationProvider(settings)

    with pytest.raises(AudioProcessingError, match="HUGGINGFACE_TOKEN") as exc_info:
        await provider.diarize(b"irrelevant-bytes-never-reached", sample_rate=16000)

    # The guard fires before any network call/model load is attempted, so
    # the pipeline attribute must remain unset - no partial state leaks out.
    assert provider._pipeline is None
    assert "irrelevant-bytes-never-reached" not in str(exc_info.value)
