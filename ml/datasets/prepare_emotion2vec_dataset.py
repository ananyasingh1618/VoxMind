#!/usr/bin/env python3
"""Builds the unified, speaker-disjoint TESS+EMO-DB training manifest for
the emotion2vec+-based production model (docs/emotion.md). Mirrors
prepare_emotion_dataset_v2.py's structure closely (same integrity checks,
reused unmodified from integrity_v2.py since the six-class target
vocabulary is identical) - does not download anything itself.

RAVDESS is deliberately NOT included here: it is used only as an external
robustness benchmark (evaluated with the existing, frozen v1 speaker-
independent test split - ml/datasets/manifests/ravdess_manifest.jsonl),
never trained on for this model, per docs/emotion.md's explicit design.

TESS's real, hard limitation (only 2 speakers total) means a true 3-way
speaker-disjoint split isn't possible for TESS alone - handled here by
assigning one TESS speaker to train and the other to validation+test
combined would still leak vocabulary/word-identity overlap across splits
in a way that's misleading; instead this script keeps TESS's 2 speakers
both in TRAIN (its lexical/lexeme diversity is what TESS contributes) and
relies on EMO-DB's 10 speakers (genuinely split 7/1/2 by speaker) for real
speaker-disjoint validation/test signal, with TESS's own severe speaker
limitation stated explicitly in the manifest report rather than silently
worked around.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from ml.datasets import emodb, tess
from ml.datasets.emotion2vec_labels import DATASET_MAPPINGS, V2_LABELS
from ml.datasets.integrity_v2 import assert_no_cross_dataset_speaker_collision, verify_and_annotate
from ml.datasets.schema import DatasetSplit, EmotionSample, label_distribution, save_manifest
from ml.datasets.splitting import assert_no_speaker_leakage, speaker_independent_split


def _namespace_speaker_id(sample: EmotionSample) -> EmotionSample:
    if sample.speaker_id is None:
        return sample
    return replace(sample, speaker_id=f"{sample.dataset_source}:{sample.speaker_id}")


def _map_to_v2_labels(samples: list[EmotionSample]) -> tuple[list[EmotionSample], dict[str, int]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tess-dir", required=True)
    parser.add_argument("--emodb-parquet", required=True)
    parser.add_argument("--emodb-audio-out", required=True, help="Where EMO-DB's embedded WAV bytes are written as real files.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()

    try:
        tess_samples = tess.build_manifest_from_directory(args.tess_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        emodb_samples = emodb.build_manifest_from_parquet(args.emodb_parquet, audio_out_dir=args.emodb_audio_out)
    except (FileNotFoundError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Raw TESS clips (excl. 'ps'): {len(tess_samples)}")
    print(f"Raw EMO-DB clips (excl. 'boredom'): {len(emodb_samples)}")

    combined_raw = tess_samples + emodb_samples
    mapped, excluded_by_protocol = _map_to_v2_labels(combined_raw)
    print(f"After label mapping: {len(mapped)} clips (excluded by protocol: {excluded_by_protocol})")

    namespaced = [_namespace_speaker_id(s) for s in mapped]
    survivors, integrity_report = verify_and_annotate(namespaced)
    if integrity_report.removed:
        print(f"Integrity check removed {len(integrity_report.removed)} sample(s):")
        for r in integrity_report.removed[:20]:
            print(f"  [{r.dataset_source}] {r.sample_id}: {r.reason}")
    else:
        print("Integrity check: no malformed files, duplicates, or out-of-vocabulary labels found.")

    for label in label_distribution(survivors):
        if label not in V2_LABELS:
            print(f"error: internal inconsistency - label {label!r} not in V2_LABELS.", file=sys.stderr)
            sys.exit(1)

    # TESS: all speakers (both of them) go to TRAIN - see module docstring.
    # EMO-DB: genuinely speaker-disjoint 3-way split (its 10 real speakers
    # give real headroom for this, unlike TESS's 2).
    tess_final = [s for s in survivors if s.dataset_source == "tess"]
    tess_split = [replace(s, split=DatasetSplit.TRAIN) for s in tess_final]

    emodb_final = [s for s in survivors if s.dataset_source == "emodb"]
    emodb_split = speaker_independent_split(
        emodb_final,
        train_fraction=1.0 - args.validation_fraction - args.test_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    split_samples = tess_split + emodb_split
    assert_no_speaker_leakage(split_samples)
    assert_no_cross_dataset_speaker_collision(split_samples)

    save_manifest(split_samples, args.output)

    report: dict = {
        "seed": args.seed,
        "raw_counts": {"tess": len(tess_samples), "emodb": len(emodb_samples)},
        "excluded_by_protocol": excluded_by_protocol,
        "integrity": integrity_report.to_dict(),
        "final_total_clips": len(split_samples),
        "overall_class_distribution": label_distribution(split_samples),
        "tess_speaker_limitation": (
            "TESS has only 2 total speakers (OAF, YAF) - both are kept in TRAIN entirely; "
            "TESS contributes lexical/word diversity, not speaker-independent validation/test "
            "signal. This is a real, documented limitation of the source dataset, not a choice "
            "that could be engineered around within this dataset alone."
        ),
        "per_dataset_per_split": {},
    }
    for dataset_source in ("tess", "emodb"):
        report["per_dataset_per_split"][dataset_source] = {}
        for split in DatasetSplit:
            subset = [s for s in split_samples if s.dataset_source == dataset_source and s.split == split]
            speakers = sorted({s.speaker_id for s in subset if s.speaker_id})
            report["per_dataset_per_split"][dataset_source][split.value] = {
                "clips": len(subset),
                "speakers": len(speakers),
                "class_distribution": label_distribution(subset),
            }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))

    print(f"\nWrote {len(split_samples)} samples to {args.output}")
    print(f"Wrote report to {args.report}")
    print(f"Overall class distribution: {label_distribution(split_samples)}")
    for dataset_source, splits in report["per_dataset_per_split"].items():
        print(f"\n{dataset_source}:")
        for split_name, info in splits.items():
            print(f"  {split_name}: {info['clips']} clips, {info['speakers']} speakers, {info['class_distribution']}")


if __name__ == "__main__":
    main()
