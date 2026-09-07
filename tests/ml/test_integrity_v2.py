from __future__ import annotations

import numpy as np
import soundfile as sf

from ml.datasets.integrity_v2 import (
    assert_no_cross_dataset_speaker_collision,
    probe_duration_and_sample_rate,
    verify_and_annotate,
)
from ml.datasets.schema import EmotionSample


def _write_wav(path, seconds: float = 1.0, sample_rate: int = 16000, amplitude: float = 0.1, seed: int = 0):
    rng = np.random.RandomState(seed)
    samples = (rng.randn(int(seconds * sample_rate)) * amplitude).astype(np.float32)
    sf.write(str(path), samples, sample_rate)


def test_probe_duration_and_sample_rate_reads_real_header(tmp_path):
    path = tmp_path / "clip.wav"
    _write_wav(path, seconds=2.0, sample_rate=16000)

    result = probe_duration_and_sample_rate(path)

    assert result is not None
    duration, sample_rate = result
    assert sample_rate == 16000
    assert abs(duration - 2.0) < 0.01


def test_probe_duration_returns_none_for_unreadable_file(tmp_path):
    path = tmp_path / "not_audio.wav"
    path.write_bytes(b"this is not a real wav file")

    assert probe_duration_and_sample_rate(path) is None


def test_verify_and_annotate_removes_malformed_file(tmp_path):
    good_path = tmp_path / "good.wav"
    _write_wav(good_path, seed=1)
    bad_path = tmp_path / "bad.wav"
    bad_path.write_bytes(b"not real audio")

    samples = [
        EmotionSample(sample_id="good", audio_path=str(good_path), label="happy", dataset_source="ravdess", speaker_id="ravdess:01"),
        EmotionSample(sample_id="bad", audio_path=str(bad_path), label="happy", dataset_source="ravdess", speaker_id="ravdess:02"),
    ]

    survivors, report = verify_and_annotate(samples)

    assert [s.sample_id for s in survivors] == ["good"]
    assert survivors[0].duration_seconds is not None
    assert survivors[0].sample_rate == 16000
    assert len(report.removed) == 1
    assert report.removed[0].sample_id == "bad"
    assert "malformed" in report.removed[0].reason


def test_verify_and_annotate_removes_duplicate_content_but_keeps_first(tmp_path):
    shared_path = tmp_path / "original.wav"
    _write_wav(shared_path, seed=42)
    duplicate_path = tmp_path / "duplicate.wav"
    duplicate_path.write_bytes(shared_path.read_bytes())  # byte-identical content, different file

    samples = [
        EmotionSample(sample_id="a_first", audio_path=str(shared_path), label="sad", dataset_source="crema_d", speaker_id="crema_d:1001"),
        EmotionSample(sample_id="b_duplicate", audio_path=str(duplicate_path), label="sad", dataset_source="crema_d", speaker_id="crema_d:1002"),
    ]

    survivors, report = verify_and_annotate(samples)

    assert [s.sample_id for s in survivors] == ["a_first"]
    assert len(report.removed) == 1
    assert report.removed[0].sample_id == "b_duplicate"
    assert "duplicate audio content" in report.removed[0].reason
    assert "a_first" in report.removed[0].reason


def test_verify_and_annotate_removes_out_of_vocabulary_label(tmp_path):
    path = tmp_path / "clip.wav"
    _write_wav(path, seed=7)

    samples = [
        EmotionSample(sample_id="s1", audio_path=str(path), label="calm", dataset_source="ravdess", speaker_id="ravdess:01"),
    ]

    survivors, report = verify_and_annotate(samples)

    assert survivors == []
    assert len(report.removed) == 1
    assert "not in V2_LABELS" in report.removed[0].reason


def test_verify_and_annotate_removes_missing_file():
    samples = [
        EmotionSample(sample_id="missing", audio_path="/nonexistent/path.wav", label="happy", dataset_source="ravdess", speaker_id="ravdess:01"),
    ]

    survivors, report = verify_and_annotate(samples)

    assert survivors == []
    assert "does not exist" in report.removed[0].reason


def test_assert_no_cross_dataset_speaker_collision_passes_for_namespaced_ids():
    samples = [
        EmotionSample(sample_id="a", audio_path="a.wav", label="happy", dataset_source="ravdess", speaker_id="ravdess:01"),
        EmotionSample(sample_id="b", audio_path="b.wav", label="happy", dataset_source="crema_d", speaker_id="crema_d:1001"),
    ]
    assert_no_cross_dataset_speaker_collision(samples)  # must not raise


def test_assert_no_cross_dataset_speaker_collision_fails_loudly_on_real_collision():
    samples = [
        EmotionSample(sample_id="a", audio_path="a.wav", label="happy", dataset_source="ravdess", speaker_id="01"),
        EmotionSample(sample_id="b", audio_path="b.wav", label="happy", dataset_source="crema_d", speaker_id="01"),
    ]
    import pytest

    with pytest.raises(AssertionError, match="Cross-dataset speaker-id collision"):
        assert_no_cross_dataset_speaker_collision(samples)
