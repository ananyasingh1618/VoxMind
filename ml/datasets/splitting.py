"""Speaker-independent train/validation/test splitting.

Splitting at the sample level (ignoring who spoke each clip) would let the
same speaker's voice appear in both training and test data - the model
could then partly learn to recognize *that speaker's* acoustic signature
rather than emotion in general, inflating test metrics in a way that
wouldn't generalize to a new speaker. This function instead assigns whole
speakers to exactly one split, so no speaker's voice is ever seen during
both training and evaluation.
"""
from __future__ import annotations

import random

from ml.datasets.schema import DatasetSplit, EmotionSample


def speaker_independent_split(
    samples: list[EmotionSample],
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> list[EmotionSample]:
    """Returns new EmotionSample instances with `split` assigned.

    Speakers (not samples) are shuffled deterministically (via `seed`) and
    allocated to train/validation/test by *speaker count* proportions. This
    assumes a roughly balanced number of samples per speaker (true for
    RAVDESS's structured recording protocol - every actor reads the same
    fixed set of statements/emotions/intensities); a dataset with wildly
    uneven per-speaker sample counts would need a sample-count-aware
    variant, not attempted here since it isn't needed for the documented
    target dataset.

    Samples with no `speaker_id` (a dataset that doesn't provide one) fall
    back to a per-sample random split - documented explicitly as a lower
    guarantee, not silently treated as equivalent to speaker-independence.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train_fraction and validation_fraction must be in (0, 1).")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must leave room for a test split.")

    rng = random.Random(seed)

    with_speaker = [s for s in samples if s.speaker_id is not None]
    without_speaker = [s for s in samples if s.speaker_id is None]

    speakers = sorted({s.speaker_id for s in with_speaker})
    rng.shuffle(speakers)

    n_speakers = len(speakers)
    n_train = round(n_speakers * train_fraction)
    n_val = round(n_speakers * validation_fraction)
    # Whatever's left goes to test - guarantees every speaker is assigned
    # exactly once even if rounding would otherwise drop one.
    speaker_split: dict[str, DatasetSplit] = {}
    for speaker in speakers[:n_train]:
        speaker_split[speaker] = DatasetSplit.TRAIN
    for speaker in speakers[n_train : n_train + n_val]:
        speaker_split[speaker] = DatasetSplit.VALIDATION
    for speaker in speakers[n_train + n_val :]:
        speaker_split[speaker] = DatasetSplit.TEST

    result = [
        EmotionSample(
            sample_id=s.sample_id,
            audio_path=s.audio_path,
            label=s.label,
            dataset_source=s.dataset_source,
            speaker_id=s.speaker_id,
            split=speaker_split[s.speaker_id],
        )
        for s in with_speaker
    ]

    if without_speaker:
        shuffled = without_speaker[:]
        rng.shuffle(shuffled)
        n_train_samples = round(len(shuffled) * train_fraction)
        n_val_samples = round(len(shuffled) * validation_fraction)
        for i, s in enumerate(shuffled):
            if i < n_train_samples:
                split = DatasetSplit.TRAIN
            elif i < n_train_samples + n_val_samples:
                split = DatasetSplit.VALIDATION
            else:
                split = DatasetSplit.TEST
            result.append(
                EmotionSample(
                    sample_id=s.sample_id,
                    audio_path=s.audio_path,
                    label=s.label,
                    dataset_source=s.dataset_source,
                    speaker_id=None,
                    split=split,
                )
            )

    return result


def assert_no_speaker_leakage(samples: list[EmotionSample]) -> None:
    """Raises AssertionError if any speaker appears in more than one split -
    a real safety check, run by the training script before it trains
    anything, not just documentation."""
    speaker_splits: dict[str, set[DatasetSplit]] = {}
    for sample in samples:
        if sample.speaker_id is None or sample.split is None:
            continue
        speaker_splits.setdefault(sample.speaker_id, set()).add(sample.split)

    leaking = {speaker: splits for speaker, splits in speaker_splits.items() if len(splits) > 1}
    if leaking:
        raise AssertionError(f"Speaker leakage detected across splits: {leaking}")
