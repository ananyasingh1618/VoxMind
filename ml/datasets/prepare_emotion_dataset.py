#!/usr/bin/env python3
"""Builds a speaker-independent train/validation/test manifest from a local
RAVDESS download. Does not download anything itself - see
ml/datasets/README.md for where to get RAVDESS and the expected directory
layout.

Usage:
    python -m ml.datasets.prepare_emotion_dataset \\
        --ravdess-dir /path/to/ravdess/Audio_Speech_Actors_01-24 \\
        --output ml/datasets/manifests/ravdess_manifest.jsonl
"""
from __future__ import annotations

import argparse
import sys

from ml.datasets.ravdess import build_manifest_from_directory
from ml.datasets.schema import label_distribution, save_manifest
from ml.datasets.splitting import assert_no_speaker_leakage, speaker_independent_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ravdess-dir", required=True, help="Path to the extracted RAVDESS audio directory.")
    parser.add_argument("--output", required=True, help="Output manifest path (JSONL).")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        samples = build_manifest_from_directory(args.ravdess_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    split_samples = speaker_independent_split(
        samples,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    assert_no_speaker_leakage(split_samples)

    save_manifest(split_samples, args.output)

    print(f"Wrote {len(split_samples)} samples to {args.output}")
    print(f"Overall label distribution: {label_distribution(split_samples)}")
    for split_name in ("train", "validation", "test"):
        subset = [s for s in split_samples if s.split and s.split.value == split_name]
        speakers = sorted({s.speaker_id for s in subset if s.speaker_id})
        print(f"  {split_name}: {len(subset)} samples, {len(speakers)} speakers, {label_distribution(subset)}")


if __name__ == "__main__":
    main()
