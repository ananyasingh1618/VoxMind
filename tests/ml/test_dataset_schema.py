from __future__ import annotations

from ml.datasets.schema import DatasetSplit, EmotionSample, label_distribution, load_manifest, save_manifest


def test_sample_round_trips_through_dict():
    sample = EmotionSample(
        sample_id="s1", audio_path="/tmp/a.wav", label="happy", dataset_source="test", speaker_id="01", split=DatasetSplit.TRAIN
    )
    restored = EmotionSample.from_dict(sample.to_dict())
    assert restored == sample


def test_sample_with_no_split_round_trips_as_none():
    sample = EmotionSample(sample_id="s1", audio_path="/tmp/a.wav", label="happy", dataset_source="test")
    restored = EmotionSample.from_dict(sample.to_dict())
    assert restored.split is None


def test_manifest_save_and_load_round_trip(tmp_path):
    samples = [
        EmotionSample(sample_id="s1", audio_path="/a.wav", label="happy", dataset_source="test", speaker_id="01", split=DatasetSplit.TRAIN),
        EmotionSample(sample_id="s2", audio_path="/b.wav", label="sad", dataset_source="test", speaker_id="02", split=DatasetSplit.TEST),
    ]
    manifest_path = tmp_path / "manifest.jsonl"

    save_manifest(samples, manifest_path)
    loaded = load_manifest(manifest_path)

    assert loaded == samples


def test_load_missing_manifest_raises_actionable_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="prepare_emotion_dataset"):
        load_manifest(tmp_path / "does_not_exist.jsonl")


def test_label_distribution_counts_real_values():
    samples = [
        EmotionSample(sample_id=f"s{i}", audio_path="x", label=label, dataset_source="test")
        for i, label in enumerate(["happy", "happy", "sad"])
    ]
    assert label_distribution(samples) == {"happy": 2, "sad": 1}
