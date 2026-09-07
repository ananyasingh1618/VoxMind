"""Real librosa-based feature extraction against real audio (Phase 2's
speech fixture) - no mocking of the signal-processing logic itself.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from voxmind.core.exceptions import AudioProcessingError
from voxmind.services.emotion.interfaces import ACOUSTIC_VECTOR_DIM
from voxmind.services.emotion.acoustic_features import LibrosaAcousticFeatureExtractor

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.asyncio
async def test_extract_real_features_from_real_speech():
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    extractor = LibrosaAcousticFeatureExtractor()

    features = await extractor.extract(wav_bytes, sample_rate=16000)

    assert 1.5 < features.duration_seconds < 4.0
    assert 0.0 <= features.voiced_fraction <= 1.0
    assert len(features.mfcc) == 13
    # Real speech should have some voiced frames with plausible pitch.
    assert features.voiced_fraction > 0.0
    assert features.pitch_hz.mean > 0.0


@pytest.mark.asyncio
async def test_feature_vector_has_documented_fixed_length():
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    extractor = LibrosaAcousticFeatureExtractor()

    features = await extractor.extract(wav_bytes, sample_rate=16000)

    assert len(features.to_vector()) == ACOUSTIC_VECTOR_DIM


@pytest.mark.asyncio
async def test_extraction_is_deterministic():
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    extractor = LibrosaAcousticFeatureExtractor()

    first = await extractor.extract(wav_bytes, sample_rate=16000)
    second = await extractor.extract(wav_bytes, sample_rate=16000)

    assert first.to_vector() == second.to_vector()


@pytest.mark.asyncio
async def test_extraction_rejects_empty_audio():
    extractor = LibrosaAcousticFeatureExtractor()
    with pytest.raises(AudioProcessingError):
        await extractor.extract(b"", sample_rate=16000)


@pytest.mark.asyncio
async def test_extraction_rejects_unreadable_bytes():
    extractor = LibrosaAcousticFeatureExtractor()
    with pytest.raises(AudioProcessingError):
        await extractor.extract(b"not a real wav file" * 10, sample_rate=16000)


@pytest.mark.asyncio
async def test_very_short_clip_has_no_pitch_track_rather_than_fabricated_one(tmp_path):
    import soundfile as sf
    import numpy as np
    import io

    # 100ms of silence - shorter than one pyin analysis frame (2048 samples
    # at 16kHz = 128ms), so pitch extraction must be skipped, not guessed.
    samples = np.zeros(1600, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, 16000, format="WAV", subtype="PCM_16")

    extractor = LibrosaAcousticFeatureExtractor()
    features = await extractor.extract(buffer.getvalue(), sample_rate=16000)

    assert features.voiced_fraction == 0.0
    assert features.pitch_hz.mean == 0.0
