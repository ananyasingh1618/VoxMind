"""Tests real ffmpeg-based decoding - no mocking of ffmpeg or soundfile.
Uses genuine audio fixtures (tests/fixtures/*, real speech synthesized via
macOS `say` and encoded with real ffmpeg/afconvert - see fixtures/README.md).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from voxmind.core.config import get_settings
from voxmind.core.exceptions import InvalidAudioError
from voxmind.services.speech.preprocessing import preprocess_audio

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.asyncio
async def test_preprocess_real_wav_produces_correct_metadata():
    raw_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    metadata, wav_bytes = await preprocess_audio(raw_bytes, settings=get_settings())

    assert metadata.sample_rate == 16000
    assert metadata.channels == 1
    assert metadata.format == "wav"
    assert 1.5 < metadata.duration_seconds < 4.0  # "the quick brown fox..." is ~2.5s
    assert len(wav_bytes) > 0


@pytest.mark.asyncio
async def test_preprocess_real_mp3_decodes_and_normalizes():
    """The source fixture is genuinely mp3-encoded (real ffmpeg encode from
    the wav fixture) - this proves ffmpeg-based decoding handles more than
    just wav passthrough."""
    raw_bytes = (FIXTURES_DIR / "hello_world.mp3").read_bytes()
    metadata, _ = await preprocess_audio(raw_bytes, settings=get_settings())

    assert metadata.sample_rate == 16000
    assert metadata.channels == 1
    assert 1.5 < metadata.duration_seconds < 4.0


@pytest.mark.asyncio
async def test_preprocess_rejects_empty_bytes():
    with pytest.raises(InvalidAudioError):
        await preprocess_audio(b"", settings=get_settings())


@pytest.mark.asyncio
async def test_preprocess_rejects_garbage_bytes():
    with pytest.raises(InvalidAudioError):
        await preprocess_audio(b"this is not audio data at all" * 20, settings=get_settings())


@pytest.mark.asyncio
async def test_preprocess_rejects_audio_exceeding_duration_limit(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "MAX_AUDIO_DURATION_SECONDS", 1)  # fixture is ~2.5s
    raw_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    with pytest.raises(InvalidAudioError):
        await preprocess_audio(raw_bytes, settings=settings)


@pytest.mark.asyncio
async def test_preprocess_does_not_mutate_original_bytes():
    """The original upload's bytes object must be untouched - preprocessing
    reads it and produces a *new* bytes object, never modifies in place."""
    raw_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    original_copy = bytes(raw_bytes)
    _, wav_bytes = await preprocess_audio(raw_bytes, settings=get_settings())

    assert raw_bytes == original_copy
    assert wav_bytes is not raw_bytes
