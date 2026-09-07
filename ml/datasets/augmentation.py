"""Conservative, training-only waveform augmentation for Emotion Model v2
(docs/emotion.md, "Data augmentation"). Every transform here is:

  - training-only (never applied during validation/test - the caller,
    ml/training/train_emotion_model_v2.py, only invokes this module for
    the training split's DataLoader),
  - deterministic given a seed (a `random.Random` instance is threaded
    through explicitly rather than relying on global RNG state, so a
    fixed seed reproduces the exact same augmented waveforms),
  - label-preserving: gain, low-level additive noise, and mild time-
    stretching change how a clip sounds acoustically without changing
    what emotion a human would perceive it as, unlike (deliberately
    excluded) pitch-shifting or extreme time-stretching, which begin to
    change perceived arousal/emotion or speaker identity.

v1's training script uses no augmentation at all (it trains a small MLP on
precomputed, frozen features - augmenting those would mean re-running
Wav2Vec2 for every augmented variant, which v1 never needed). This module
is v2-only.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from librosa.effects import time_stretch as librosa_time_stretch


@dataclass
class AugmentationConfig:
    """Every field is a real knob read by `apply_augmentation` - nothing
    hardcoded elsewhere. Conservative bounds per docs/emotion.md's explicit
    instruction not to use extreme augmentation."""

    gain_db_range: tuple[float, float] = (-3.0, 3.0)
    noise_probability: float = 0.5
    noise_snr_db_range: tuple[float, float] = (20.0, 35.0)  # high SNR = low-level noise only
    time_stretch_probability: float = 0.3
    time_stretch_rate_range: tuple[float, float] = (0.95, 1.05)  # +/-5%, mild
    gain_probability: float = 0.5


def _apply_gain(samples: np.ndarray, rng: random.Random, config: AugmentationConfig) -> np.ndarray:
    gain_db = rng.uniform(*config.gain_db_range)
    factor = 10.0 ** (gain_db / 20.0)
    return samples * factor


def _apply_noise(samples: np.ndarray, rng: random.Random, config: AugmentationConfig) -> np.ndarray:
    snr_db = rng.uniform(*config.noise_snr_db_range)
    signal_power = float(np.mean(samples**2)) if samples.size else 0.0
    if signal_power <= 0.0:
        return samples
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    # np.random (not `rng`, a plain random.Random) is used only for the
    # per-sample Gaussian draw, seeded once from `rng` so the whole
    # pipeline is still fully deterministic for a given seed.
    noise_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    noise = noise_rng.normal(0.0, np.sqrt(noise_power), size=samples.shape).astype(np.float32)
    return samples + noise


def _apply_time_stretch(samples: np.ndarray, rng: random.Random, config: AugmentationConfig) -> np.ndarray:
    rate = rng.uniform(*config.time_stretch_rate_range)
    if samples.size < 2:
        return samples
    stretched = librosa_time_stretch(samples.astype(np.float32), rate=rate)
    return stretched.astype(np.float32)


def apply_augmentation(samples: np.ndarray, *, rng: random.Random, config: AugmentationConfig) -> np.ndarray:
    """Applies each enabled augmentation independently at its own
    probability, in a fixed order (gain -> noise -> time-stretch) so a
    given seed always produces the same sequence of random draws
    regardless of which ones happen to fire. Returns a new array; never
    mutates `samples` in place."""
    result = np.array(samples, dtype=np.float32, copy=True)

    if rng.random() < config.gain_probability:
        result = _apply_gain(result, rng, config)
    if rng.random() < config.noise_probability:
        result = _apply_noise(result, rng, config)
    if rng.random() < config.time_stretch_probability:
        result = _apply_time_stretch(result, rng, config)

    # Gain/noise can push samples outside [-1, 1] - clip rather than let it
    # silently overflow when later converted for the feature extractor.
    return np.clip(result, -1.0, 1.0)
