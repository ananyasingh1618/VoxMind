"""Verifies the REAL CREMA-D dataset when it's present on disk - mirrors
test_real_ravdess_dataset.py exactly. The dataset (~1GB of real audio,
downloaded via git-lfs from the official CheyneyComputerScience/CREMA-D
GitHub repository, ODbL-licensed, no registration required) is not
committed to this repository - see ml/datasets/README.md. This test is
skipped, not faked, on any machine without it at the documented path.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ml.datasets.crema_d import build_manifest_from_directory
from ml.datasets.schema import label_distribution
from ml.datasets.splitting import assert_no_speaker_leakage, speaker_independent_split

CREMA_D_DIR = Path(
    os.environ.get("CREMA_D_DIR", "/Users/ananyasingh/voxmind-datasets/crema-d/repo/AudioWAV")
)


def _real_files_present() -> bool:
    """Pointer (unpulled LFS) files are 130 bytes - only count this dataset
    as "present" once real audio content has actually been pulled, so this
    test is skipped (not silently wrong) on a checkout that has the repo
    but hasn't run `git lfs pull` yet."""
    if not CREMA_D_DIR.exists():
        return False
    real_files = [p for p in CREMA_D_DIR.glob("*.wav") if p.stat().st_size > 1024]
    return len(real_files) >= 7000


pytestmark = pytest.mark.skipif(
    not _real_files_present(),
    reason=(
        f"Real CREMA-D audio content not found at {CREMA_D_DIR} - see ml/datasets/README.md "
        "(git-lfs clone, ODbL license, no auth required) or set CREMA_D_DIR."
    ),
)


def test_real_dataset_parses_with_zero_malformed_samples():
    samples = build_manifest_from_directory(CREMA_D_DIR)

    assert len(samples) == 7442  # documented CREMA-D total, verified against the real download

    speakers = {s.speaker_id for s in samples}
    assert len(speakers) == 91


def test_real_dataset_class_distribution_matches_documented_crema_d_protocol():
    samples = build_manifest_from_directory(CREMA_D_DIR)
    distribution = label_distribution(samples)

    # Verified against the real filenames in this environment (session
    # inventory): 1271 clips each for angry/disgust/fear/happy/sad, 1087
    # for neutral - CREMA-D's real, documented class-count asymmetry.
    expected = {
        "angry": 1271,
        "disgust": 1271,
        "fear": 1271,
        "happy": 1271,
        "neutral": 1087,
        "sad": 1271,
    }
    assert distribution == expected


def test_real_dataset_speaker_independent_split_has_no_leakage():
    samples = build_manifest_from_directory(CREMA_D_DIR)

    split_samples = speaker_independent_split(samples, seed=42)

    assert_no_speaker_leakage(split_samples)  # must not raise
    train_speakers = {s.speaker_id for s in split_samples if s.split and s.split.value == "train"}
    val_speakers = {s.speaker_id for s in split_samples if s.split and s.split.value == "validation"}
    test_speakers = {s.speaker_id for s in split_samples if s.split and s.split.value == "test"}
    assert len(train_speakers) == 64  # round(91 * 0.7)
    assert len(val_speakers) == 14  # round(91 * 0.15)
    assert len(test_speakers) == 13  # remainder
    assert not (train_speakers & val_speakers & test_speakers)
