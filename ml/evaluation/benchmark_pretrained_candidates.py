#!/usr/bin/env python3
"""Benchmarks external pretrained speech-emotion-recognition candidates
against VoxMind's own, real, speaker-independent RAVDESS held-out test
split (the same 180-clip / 3-speaker split `ravdess-v1` was evaluated on -
`ml/datasets/manifests/ravdess_manifest.jsonl`, split="test"). Never
retrains anything - loads each candidate's public pretrained checkpoint
as-is and runs real inference.

For candidates whose `config.json` `id2label` is undocumented/generic
(e.g. "LABEL_0".."LABEL_7"), this script evaluates every plausible label
ORDERING against our real ground truth and reports the best-scoring one
explicitly labeled as "assumed / reverse-engineered - not officially
documented by the model author" - never silently presented as a confirmed
mapping.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.datasets.schema import DatasetSplit, load_manifest
from ml.evaluation.metrics import compute_classification_metrics

RAVDESS_CANONICAL_ORDER = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
ALPHABETICAL_ORDER = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


def load_test_clips(manifest_path: str) -> tuple[list[str], list[str]]:
    samples = load_manifest(manifest_path)
    test_samples = [s for s in samples if s.split == DatasetSplit.TEST]
    return [s.audio_path for s in test_samples], [s.label for s in test_samples]


@torch.no_grad()
def run_inference(repo_id: str, audio_paths: list[str], device: str = "cpu") -> list[int]:
    """Returns raw predicted class indices (0..num_labels-1) - label-name
    interpretation happens separately, since for some candidates that
    mapping itself is unknown/must be evaluated as a hypothesis."""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(repo_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(repo_id)
    model.to(device)
    model.eval()

    predictions = []
    for i, path in enumerate(audio_paths):
        samples, sr = sf.read(path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sr != feature_extractor.sampling_rate:
            import librosa

            samples = librosa.resample(samples, orig_sr=sr, target_sr=feature_extractor.sampling_rate)
        inputs = feature_extractor(samples, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        logits = model(inputs["input_values"].to(device)).logits
        predictions.append(int(logits.argmax(dim=-1).item()))
        if (i + 1) % 30 == 0 or i == len(audio_paths) - 1:
            print(f"  {i + 1}/{len(audio_paths)}")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--manifest", default="ml/datasets/manifests/ravdess_manifest.jsonl")
    parser.add_argument(
        "--label-order",
        choices=["config", "ravdess_canonical", "alphabetical", "try_all"],
        default="config",
        help="'config' trusts the model's own id2label (use only if it has real names). "
        "'try_all' evaluates every plausible ordering against our ground truth and reports each.",
    )
    parser.add_argument("--out", default=None, help="Optional path to write full results JSON.")
    args = parser.parse_args()

    audio_paths, y_true = load_test_clips(args.manifest)
    print(f"Real RAVDESS held-out test set: {len(audio_paths)} clips, "
          f"speakers={sorted({Path(p).stem.split('-')[-1] for p in audio_paths})}")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.repo_id)
    config_id2label = [config.id2label[i] for i in range(len(config.id2label))]
    print(f"Model's own config.json id2label: {config_id2label}")

    print(f"\nRunning real inference with {args.repo_id} on {len(audio_paths)} real audio clips...")
    pred_indices = run_inference(args.repo_id, audio_paths)

    label_names = sorted(set(y_true))

    orderings_to_try: dict[str, list[str]]
    if args.label_order == "config":
        orderings_to_try = {"config (model's own id2label)": config_id2label}
    elif args.label_order == "ravdess_canonical":
        orderings_to_try = {"ravdess_canonical (ASSUMED)": RAVDESS_CANONICAL_ORDER}
    elif args.label_order == "alphabetical":
        orderings_to_try = {"alphabetical (ASSUMED)": ALPHABETICAL_ORDER}
    else:
        orderings_to_try = {
            "config (model's own id2label)": config_id2label,
            "ravdess_canonical (ASSUMED)": RAVDESS_CANONICAL_ORDER,
            "alphabetical (ASSUMED)": ALPHABETICAL_ORDER,
        }

    results = {}
    for ordering_name, ordering in orderings_to_try.items():
        if any(label.startswith("LABEL_") for label in ordering):
            print(f"\n=== Ordering: {ordering_name} - SKIPPED (undocumented, no real names) ===")
            continue
        y_pred = [ordering[i] for i in pred_indices]
        metrics = compute_classification_metrics(y_true, y_pred, label_names)
        print(f"\n=== Ordering: {ordering_name} ===")
        print(f"accuracy={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f} "
              f"weighted_f1={metrics.weighted_f1:.4f} balanced_accuracy={metrics.balanced_accuracy:.4f}")
        results[ordering_name] = metrics.to_dict()

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote full results to {args.out}")


if __name__ == "__main__":
    main()
