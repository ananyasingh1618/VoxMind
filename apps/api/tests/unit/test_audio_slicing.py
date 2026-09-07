from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf
import io

from voxmind.core.exceptions import InvalidAudioError
from voxmind.services.emotion.audio_slicing import slice_wav_segment

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_slice_returns_correct_duration():
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()

    segment_bytes, sample_rate = slice_wav_segment(wav_bytes, start_ms=0, end_ms=1000)

    with sf.SoundFile(io.BytesIO(segment_bytes)) as f:
        duration_ms = round(len(f) / f.samplerate * 1000)
    assert abs(duration_ms - 1000) <= 5  # frame-rounding tolerance
    assert sample_rate == 16000


def test_slice_clamps_end_beyond_audio_length():
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()

    segment_bytes, _ = slice_wav_segment(wav_bytes, start_ms=0, end_ms=999_999)

    with sf.SoundFile(io.BytesIO(segment_bytes)) as f:
        full_duration_ms = round(len(f) / f.samplerate * 1000)
    original_duration_ms = round(
        sf.info(io.BytesIO(wav_bytes)).duration * 1000
    )
    assert abs(full_duration_ms - original_duration_ms) <= 5


def test_slice_rejects_empty_range():
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()

    with pytest.raises(InvalidAudioError):
        slice_wav_segment(wav_bytes, start_ms=1000, end_ms=1000)


def test_slice_rejects_out_of_range_start():
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()

    with pytest.raises(InvalidAudioError):
        slice_wav_segment(wav_bytes, start_ms=999_999, end_ms=1_000_999)
