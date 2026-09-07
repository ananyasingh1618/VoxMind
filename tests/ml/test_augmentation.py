from __future__ import annotations

import random

import numpy as np

from ml.datasets.augmentation import AugmentationConfig, apply_augmentation


def _sine_wave(seconds: float = 1.0, sample_rate: int = 16000, freq: float = 220.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_same_seed_produces_identical_augmented_output():
    waveform = _sine_wave()
    config = AugmentationConfig()

    out1 = apply_augmentation(waveform, rng=random.Random(42), config=config)
    out2 = apply_augmentation(waveform, rng=random.Random(42), config=config)

    assert np.array_equal(out1, out2)


def test_different_seeds_produce_different_output():
    waveform = _sine_wave()
    config = AugmentationConfig()

    out1 = apply_augmentation(waveform, rng=random.Random(1), config=config)
    out2 = apply_augmentation(waveform, rng=random.Random(2), config=config)

    assert not np.array_equal(out1, out2)


def test_output_never_exceeds_valid_waveform_range():
    waveform = _sine_wave(freq=110.0) * 3.0  # deliberately already out-of-range input
    config = AugmentationConfig(gain_db_range=(6.0, 6.0), gain_probability=1.0, noise_probability=0.0, time_stretch_probability=0.0)

    out = apply_augmentation(waveform, rng=random.Random(0), config=config)

    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_never_mutates_input_in_place():
    waveform = _sine_wave()
    original = waveform.copy()
    config = AugmentationConfig()

    apply_augmentation(waveform, rng=random.Random(0), config=config)

    assert np.array_equal(waveform, original)


def test_disabling_all_augmentations_still_returns_a_valid_waveform():
    waveform = _sine_wave()
    config = AugmentationConfig(gain_probability=0.0, noise_probability=0.0, time_stretch_probability=0.0)

    out = apply_augmentation(waveform, rng=random.Random(0), config=config)

    assert np.allclose(out, waveform, atol=1e-6)
