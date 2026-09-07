"""Genuine pyannote.audio diarization execution against the real, gated
`pyannote/speaker-diarization-3.1` model. Requires a Hugging Face account
that has accepted that model's (and its `pyannote/segmentation-3.0`
dependency's) license terms, plus a read-scoped access token set as
HUGGINGFACE_TOKEN - see docs/audio.md for the exact steps.

This test is SKIPPED in this repository's current environment because no
such token is configured - that is reported honestly rather than faked.
Set HUGGINGFACE_TOKEN and re-run with: pytest -m requires_hf_token
"""
from __future__ import annotations

from pathlib import Path

import pytest

from voxmind.core.config import get_settings
from voxmind.services.speech.diarization_provider import PyannoteDiarizationProvider

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.requires_hf_token
@pytest.mark.asyncio
@pytest.mark.skipif(
    not get_settings().HUGGINGFACE_TOKEN,
    reason=(
        "HUGGINGFACE_TOKEN is not configured in this environment - real pyannote "
        "diarization requires a Hugging Face account that has accepted the gated "
        "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 model terms, "
        "plus a read-scoped access token. See docs/audio.md."
    ),
)
async def test_real_diarization_runs_against_genuine_audio():
    settings = get_settings()
    provider = PyannoteDiarizationProvider(settings)

    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    result = await provider.diarize(wav_bytes, sample_rate=16000)

    assert result.model_version == settings.DIARIZATION_MODEL
    assert result.num_speakers >= 1
    for segment in result.speaker_segments:
        assert segment.speaker_label.startswith("speaker_")
