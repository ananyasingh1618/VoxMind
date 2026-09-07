from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from ml.datasets.emodb import EMOTION_INT_TO_LABEL, SPEAKER_DEMOGRAPHICS, build_manifest_from_parquet


def _fake_wav_bytes(seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    samples = (np.random.RandomState(0).randn(int(seconds * sample_rate)) * 0.05).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


def test_emotion_int_mapping_has_all_seven_documented_classes():
    assert EMOTION_INT_TO_LABEL == {
        0: "anger", 1: "boredom", 2: "disgust", 3: "fear", 4: "happiness", 5: "neutral", 6: "sadness",
    }


def test_speaker_demographics_table_has_exactly_ten_unique_speakers():
    assert len(SPEAKER_DEMOGRAPHICS) == 10
    assert len(set(SPEAKER_DEMOGRAPHICS.values())) == 10


def test_build_manifest_excludes_boredom_and_maps_speaker_id(tmp_path):
    rows = [
        {"age": 31.0, "gender": 1, "emotion": 0, "audio": {"bytes": _fake_wav_bytes()}},  # speaker 03, anger
        {"age": 31.0, "gender": 1, "emotion": 1, "audio": {"bytes": _fake_wav_bytes()}},  # speaker 03, boredom - excluded
        {"age": 21.0, "gender": 0, "emotion": 4, "audio": {"bytes": _fake_wav_bytes()}},  # speaker 09, happiness
    ]
    parquet_path = tmp_path / "test.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path)

    samples = build_manifest_from_parquet(parquet_path, audio_out_dir=tmp_path / "audio")

    assert len(samples) == 2
    labels_by_speaker = {s.speaker_id: s.label for s in samples}
    assert labels_by_speaker == {"03": "anger", "09": "happiness"}
    assert all(s.dataset_source == "emodb" for s in samples)
    assert all(Path(s.audio_path).exists() for s in samples)


def test_build_manifest_raises_on_undocumented_demographics(tmp_path):
    rows = [{"age": 999.0, "gender": 1, "emotion": 0, "audio": {"bytes": _fake_wav_bytes()}}]
    parquet_path = tmp_path / "test.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path)

    with pytest.raises(AssertionError, match="not in the documented"):
        build_manifest_from_parquet(parquet_path, audio_out_dir=tmp_path / "audio")
