"""Verifies the REAL RAVDESS dataset when it's present on disk - genuinely
parses all 1440 real speech files, not a synthetic stand-in. The dataset
itself (~200MB) is not committed to this repository (see
ml/datasets/README.md for the public Zenodo download), so this test is
skipped, not faked, on any machine that doesn't have it at the documented
path. `tests/ml/test_ravdess_parser.py` already covers the parser's logic
exhaustively against small synthetic-but-convention-accurate filenames;
this file additionally proves it against the real corpus end-to-end.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ml.datasets.ravdess import build_manifest_from_directory
from ml.datasets.schema import label_distribution
from ml.datasets.splitting import assert_no_speaker_leakage, speaker_independent_split

RAVDESS_DIR = Path(
    os.environ.get("RAVDESS_DIR", "/Users/ananyasingh/voxmind-datasets/ravdess/extracted")
)

pytestmark = pytest.mark.skipif(
    not RAVDESS_DIR.exists(),
    reason=(
        f"Real RAVDESS dataset not found at {RAVDESS_DIR} - download it per "
        "ml/datasets/README.md (public, no auth required) or set RAVDESS_DIR."
    ),
)


def test_real_dataset_parses_with_zero_malformed_samples():
    samples = build_manifest_from_directory(RAVDESS_DIR)

    # RAVDESS's speech-only release is exactly 24 actors x 60 clips.
    assert len(samples) == 1440

    speakers = {s.speaker_id for s in samples}
    assert len(speakers) == 24
    assert speakers == {f"{i:02d}" for i in range(1, 25)}


def test_real_dataset_class_distribution_matches_documented_ravdess_protocol():
    samples = build_manifest_from_directory(RAVDESS_DIR)
    distribution = label_distribution(samples)

    # 24 actors x 2 statements x 2 repetitions x 2 intensities = 192 for the
    # 7 emotions with both intensities; "neutral" has only the normal
    # intensity (no "strong neutral" in RAVDESS's protocol) => 96.
    expected = {
        "angry": 192,
        "calm": 192,
        "disgust": 192,
        "fearful": 192,
        "happy": 192,
        "neutral": 96,
        "sad": 192,
        "surprised": 192,
    }
    assert distribution == expected


def test_real_dataset_speaker_independent_split_has_no_leakage():
    samples = build_manifest_from_directory(RAVDESS_DIR)

    split_samples = speaker_independent_split(samples, seed=42)

    assert_no_speaker_leakage(split_samples)  # must not raise
    train_speakers = {s.speaker_id for s in split_samples if s.split and s.split.value == "train"}
    val_speakers = {s.speaker_id for s in split_samples if s.split and s.split.value == "validation"}
    test_speakers = {s.speaker_id for s in split_samples if s.split and s.split.value == "test"}
    assert len(train_speakers) == 17
    assert len(val_speakers) == 4
    assert len(test_speakers) == 3
    assert not (train_speakers & val_speakers & test_speakers)
