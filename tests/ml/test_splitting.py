from __future__ import annotations

import pytest

from ml.datasets.schema import DatasetSplit, EmotionSample
from ml.datasets.splitting import assert_no_speaker_leakage, speaker_independent_split


def _make_samples(n_speakers: int, samples_per_speaker: int) -> list[EmotionSample]:
    samples = []
    labels = ["happy", "sad", "angry", "neutral"]
    for speaker_idx in range(n_speakers):
        for i in range(samples_per_speaker):
            samples.append(
                EmotionSample(
                    sample_id=f"speaker{speaker_idx}_sample{i}",
                    audio_path=f"/fake/{speaker_idx}_{i}.wav",
                    label=labels[i % len(labels)],
                    dataset_source="synthetic-test",
                    speaker_id=f"speaker_{speaker_idx}",
                )
            )
    return samples


def test_every_speaker_is_assigned_to_exactly_one_split():
    samples = _make_samples(n_speakers=20, samples_per_speaker=5)

    result = speaker_independent_split(samples, seed=1)

    assert_no_speaker_leakage(result)  # must not raise
    assert all(s.split is not None for s in result)


def test_split_is_deterministic_given_the_same_seed():
    samples = _make_samples(n_speakers=20, samples_per_speaker=5)

    result_a = speaker_independent_split(samples, seed=7)
    result_b = speaker_independent_split(samples, seed=7)

    assignments_a = {s.sample_id: s.split for s in result_a}
    assignments_b = {s.sample_id: s.split for s in result_b}
    assert assignments_a == assignments_b


def test_different_seeds_can_produce_different_splits():
    samples = _make_samples(n_speakers=20, samples_per_speaker=5)

    result_a = speaker_independent_split(samples, seed=1)
    result_b = speaker_independent_split(samples, seed=999)

    assignments_a = {s.sample_id: s.split for s in result_a}
    assignments_b = {s.sample_id: s.split for s in result_b}
    assert assignments_a != assignments_b


def test_split_proportions_are_approximately_correct_by_speaker_count():
    samples = _make_samples(n_speakers=20, samples_per_speaker=5)

    result = speaker_independent_split(samples, train_fraction=0.7, validation_fraction=0.15, seed=1)

    train_speakers = {s.speaker_id for s in result if s.split == DatasetSplit.TRAIN}
    val_speakers = {s.speaker_id for s in result if s.split == DatasetSplit.VALIDATION}
    test_speakers = {s.speaker_id for s in result if s.split == DatasetSplit.TEST}

    assert len(train_speakers) == 14  # round(20 * 0.7)
    assert len(val_speakers) == 3  # round(20 * 0.15)
    assert len(test_speakers) == 3
    assert train_speakers | val_speakers | test_speakers == {s.speaker_id for s in samples}


def test_detects_actual_speaker_leakage():
    samples = _make_samples(n_speakers=2, samples_per_speaker=2)
    # Manually construct a leaking assignment: same speaker in two splits.
    leaking = [
        EmotionSample(
            sample_id=s.sample_id,
            audio_path=s.audio_path,
            label=s.label,
            dataset_source=s.dataset_source,
            speaker_id=s.speaker_id,
            split=DatasetSplit.TRAIN if i == 0 else DatasetSplit.TEST,
        )
        for i, s in enumerate(samples)
        if s.speaker_id == "speaker_0"
    ]

    with pytest.raises(AssertionError, match="Speaker leakage"):
        assert_no_speaker_leakage(leaking)


def test_invalid_fractions_are_rejected():
    samples = _make_samples(n_speakers=5, samples_per_speaker=2)

    with pytest.raises(ValueError):
        speaker_independent_split(samples, train_fraction=0.8, validation_fraction=0.3)


def test_samples_without_speaker_id_use_documented_random_fallback():
    samples = [
        EmotionSample(sample_id=f"s{i}", audio_path=f"/{i}.wav", label="happy", dataset_source="test", speaker_id=None)
        for i in range(10)
    ]

    result = speaker_independent_split(samples, seed=1)

    assert all(s.split is not None for s in result)
    assert len(result) == 10
