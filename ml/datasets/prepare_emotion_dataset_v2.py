#!/usr/bin/env python3
"""Builds Emotion Model v2's unified, speaker-independent RAVDESS+CREMA-D
manifest (docs/emotion.md, "Emotion Model v2"). Does not download anything
itself - see ml/datasets/README.md.

Pipeline (every step is real, nothing here is invented or assumed):
  1. Parse both raw datasets (ravdess.py, crema_d.py).
  2. Map each dataset's raw labels onto the v2 six-class vocabulary
     (emotion_v2_labels.py), dropping RAVDESS's "calm"/"surprised" clips -
     never reinterpreting them as another class.
  3. Namespace every speaker_id as f"{dataset_source}:{raw_id}" - makes a
     cross-dataset speaker-id collision structurally impossible (RAVDESS
     uses "01".."24", CREMA-D uses "1001".."1091" - no numeric collision
     exists in this corpus pairing, but namespacing removes any reliance on
     that coincidence).
  4. Run every required integrity check (integrity_v2.py): malformed/
     unreadable files, duplicate audio content, duplicate sample ids, every
     label in-vocabulary, every file exists. Anything removed is reported,
     never silently dropped.
  5. Split **each dataset's speakers independently** (not one combined
     shuffle) into train/validation/test via the existing, unmodified
     speaker_independent_split() - so both datasets are represented in
     every split in roughly their overall proportion, rather than risking
     (by chance, on a single combined shuffle) one split becoming almost
     entirely one dataset. This reuses splitting.py exactly as RAVDESS-only
     preparation already does, just called twice.
  6. Verify no speaker leakage (assert_no_speaker_leakage, unchanged) and no
     cross-dataset id collision (defensive re-check).
  7. Write the manifest (JSONL, reusing save_manifest) plus a JSON stats
     report (per-dataset/per-split counts, class distribution, speaker
     counts, and everything integrity-removed and why).

Usage:
    python -m ml.datasets.prepare_emotion_dataset_v2 \\
        --ravdess-dir /path/to/ravdess/Audio_Speech_Actors_01-24 \\
        --crema-d-dir /path/to/crema-d/AudioWAV \\
        --output ml/datasets/manifests/emotion_v2_manifest.jsonl \\
        --report ml/datasets/manifests/emotion_v2_manifest.stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from ml.datasets import crema_d, ravdess
from ml.datasets.emotion_v2_labels import DATASET_MAPPINGS, V2_LABELS
from ml.datasets.integrity_v2 import assert_no_cross_dataset_speaker_collision, verify_and_annotate
from ml.datasets.schema import DatasetSplit, EmotionSample, label_distribution, save_manifest
from ml.datasets.splitting import assert_no_speaker_leakage, speaker_independent_split


def _namespace_speaker_id(sample: EmotionSample) -> EmotionSample:
    if sample.speaker_id is None:
        return sample
    return replace(sample, speaker_id=f"{sample.dataset_source}:{sample.speaker_id}")


def _map_to_v2_labels(samples: list[EmotionSample]) -> tuple[list[EmotionSample], dict[str, int]]:
    """Applies each dataset's documented raw->v2 label mapping, dropping
    samples that map to None. Returns the mapped samples plus a count of
    how many were excluded per raw label, for honest reporting - excluded
    clips are a deliberate, documented protocol choice (RAVDESS's calm/
    surprised), not a data-quality removal, so they're reported separately
    from integrity_v2.py's "removed" list."""
    mapped: list[EmotionSample] = []
    excluded_counts: dict[str, int] = {}
    for sample in samples:
        mapping = DATASET_MAPPINGS[sample.dataset_source]
        v2_label = mapping.get(sample.label)
        if v2_label is None:
            key = f"{sample.dataset_source}:{sample.label}"
            excluded_counts[key] = excluded_counts.get(key, 0) + 1
            continue
        mapped.append(replace(sample, label=v2_label))
    return mapped, excluded_counts


def _split_per_dataset(
    samples: list[EmotionSample], *, train_fraction: float, validation_fraction: float, seed: int
) -> list[EmotionSample]:
    """Splits each dataset_source's speakers independently (see module
    docstring point 5) then concatenates. Reuses speaker_independent_split
    unmodified - called once per dataset rather than once for everything."""
    result: list[EmotionSample] = []
    for dataset_source in sorted({s.dataset_source for s in samples}):
        subset = [s for s in samples if s.dataset_source == dataset_source]
        result.extend(
            speaker_independent_split(
                subset, train_fraction=train_fraction, validation_fraction=validation_fraction, seed=seed
            )
        )
    return result


def _speaker_counts_by_split(samples: list[EmotionSample]) -> dict:
    report: dict = {}
    for dataset_source in sorted({s.dataset_source for s in samples}):
        report[dataset_source] = {}
        for split in DatasetSplit:
            subset = [s for s in samples if s.dataset_source == dataset_source and s.split == split]
            speakers = sorted({s.speaker_id for s in subset if s.speaker_id})
            report[dataset_source][split.value] = {
                "clips": len(subset),
                "speakers": len(speakers),
                "class_distribution": label_distribution(subset),
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ravdess-dir", required=True)
    parser.add_argument("--crema-d-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True, help="Where to write the JSON stats/integrity report.")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        ravdess_samples = ravdess.build_manifest_from_directory(args.ravdess_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        crema_d_samples = crema_d.build_manifest_from_directory(args.crema_d_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Raw RAVDESS clips (speech only): {len(ravdess_samples)}")
    print(f"Raw CREMA-D clips: {len(crema_d_samples)}")

    combined_raw = ravdess_samples + crema_d_samples
    mapped, excluded_by_protocol = _map_to_v2_labels(combined_raw)
    print(f"After six-class label mapping: {len(mapped)} clips "
          f"(excluded by protocol - not a data-quality issue: {excluded_by_protocol})")

    namespaced = [_namespace_speaker_id(s) for s in mapped]

    survivors, integrity_report = verify_and_annotate(namespaced)
    if integrity_report.removed:
        print(f"Integrity check removed {len(integrity_report.removed)} sample(s):")
        for r in integrity_report.removed[:20]:
            print(f"  [{r.dataset_source}] {r.sample_id}: {r.reason}")
        if len(integrity_report.removed) > 20:
            print(f"  ... and {len(integrity_report.removed) - 20} more (see the report file).")
    else:
        print("Integrity check: no malformed files, duplicates, or out-of-vocabulary labels found.")

    for label in label_distribution(survivors):
        if label not in V2_LABELS:
            print(f"error: internal inconsistency - label {label!r} survived integrity check but is not in V2_LABELS.", file=sys.stderr)
            sys.exit(1)

    split_samples = _split_per_dataset(
        survivors,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    assert_no_speaker_leakage(split_samples)
    assert_no_cross_dataset_speaker_collision(split_samples)

    save_manifest(split_samples, args.output)

    speaker_report = _speaker_counts_by_split(split_samples)
    report_payload = {
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "raw_counts": {"ravdess": len(ravdess_samples), "crema_d": len(crema_d_samples)},
        "excluded_by_protocol": excluded_by_protocol,
        "integrity": integrity_report.to_dict(),
        "final_total_clips": len(split_samples),
        "overall_class_distribution": label_distribution(split_samples),
        "per_dataset_per_split": speaker_report,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report_payload, indent=2))

    print(f"\nWrote {len(split_samples)} samples to {args.output}")
    print(f"Wrote integrity/stats report to {args.report}")
    print(f"Overall class distribution: {label_distribution(split_samples)}")
    for dataset_source, splits in speaker_report.items():
        print(f"\n{dataset_source}:")
        for split_name, info in splits.items():
            print(f"  {split_name}: {info['clips']} clips, {info['speakers']} speakers, {info['class_distribution']}")


if __name__ == "__main__":
    main()
