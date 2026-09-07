#!/usr/bin/env python3
"""Independently re-evaluates an already-registered Emotion Model v2
version against a manifest split - the v2 analog of
evaluate_emotion_model.py, adapted for v2's raw-waveform input (no
precomputed embedding to just re-run a classifier against - the whole
fine-tuned backbone runs again here) and its per-dataset-source breakdown.
Reuses the same `compute_classification_metrics` v1's script and
train_emotion_model_v2.py both use - one metrics implementation, not three.

Usage:
    python -m ml.evaluation.evaluate_emotion_model_v2 \\
        --model-version-id <uuid> \\
        --manifest ml/datasets/manifests/emotion_v2_manifest.jsonl \\
        --split test
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

from ml.datasets.augmentation import AugmentationConfig
from ml.datasets.schema import DatasetSplit, load_manifest
from ml.training.train_emotion_model_v2 import (
    RawWaveformDataset,
    evaluate,
    make_collate_fn,
    preprocess_and_cache_waveforms,
    resolve_device,
)
from voxmind.core.config import get_settings
from voxmind.services.emotion.v2.model import EmotionCheckpointV2
from voxmind.services.storage.factory import build_storage_backend


async def run_evaluation(args: argparse.Namespace) -> None:
    from voxmind.db.session import AsyncSessionLocal
    from voxmind.repositories.evaluation_run_repository import EvaluationRunRepository
    from voxmind.repositories.model_version_repository import ModelVersionRepository

    settings = get_settings()
    device = resolve_device(args.device)

    async with AsyncSessionLocal() as session:
        repo = ModelVersionRepository(session)
        model_version = await repo.get_by_id(uuid.UUID(args.model_version_id))
        if model_version is None:
            print(f"error: no model_versions row with id {args.model_version_id}", file=sys.stderr)
            sys.exit(1)
        if not model_version.artifact_storage_key:
            print("error: this model version has no artifact_storage_key (was it ever trained?).", file=sys.stderr)
            sys.exit(1)
        if (model_version.training_config or {}).get("architecture") != "v2-wav2vec2-attention":
            print("error: this model version is not a v2 (wav2vec2-attention) model - use evaluate_emotion_model.py for v1.", file=sys.stderr)
            sys.exit(1)

        storage = build_storage_backend(settings)
        checkpoint_bytes = await storage.download(model_version.artifact_storage_key)
        checkpoint = EmotionCheckpointV2.from_bytes(checkpoint_bytes)
        model = checkpoint.build_model()
        model.to(device)

        samples = load_manifest(args.manifest)
        target_split = DatasetSplit(args.split)
        eval_samples = [s for s in samples if s.split == target_split]
        if not eval_samples:
            print(f"error: manifest has no samples in split {args.split!r}.", file=sys.stderr)
            sys.exit(1)

        cache_dir = Path(args.waveform_cache_dir)
        print(f"Preprocessing {len(eval_samples)} clips (cached to {cache_dir})...")
        await preprocess_and_cache_waveforms(eval_samples, settings=settings, cache_dir=cache_dir)
        dataset = RawWaveformDataset(
            eval_samples, cache_dir=cache_dir, label_names=checkpoint.config.label_names, augment=False,
            augmentation_config=AugmentationConfig(), seed=0, max_audio_seconds=args.max_audio_seconds,
        )
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint.config.base_model)
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=make_collate_fn(feature_extractor))

        metrics = evaluate(model, loader, checkpoint.config.label_names, device)

        print(f"\n=== Independent v2 evaluation: model {model_version.id} ({model_version.version_tag}) ===")
        print(f"split={args.split} n={metrics.n_samples}")
        print(f"accuracy={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f} "
              f"weighted_f1={metrics.weighted_f1:.4f} balanced_accuracy={metrics.balanced_accuracy:.4f}")
        print("per-class:")
        for label, m in metrics.per_class.items():
            print(f"  {label}: precision={m.precision:.3f} recall={m.recall:.3f} f1={m.f1:.3f} support={m.support}")
        print(f"confusion_matrix (rows=true, cols=pred, order={metrics.label_order}):")
        for row in metrics.confusion_matrix:
            print(f"  {row}")

        per_dataset: dict[str, dict] = {}
        for dataset_source in sorted({s.dataset_source for s in eval_samples}):
            subset_samples = [s for s in eval_samples if s.dataset_source == dataset_source]
            subset_ds = RawWaveformDataset(
                subset_samples, cache_dir=cache_dir, label_names=checkpoint.config.label_names, augment=False,
                augmentation_config=AugmentationConfig(), seed=0, max_audio_seconds=args.max_audio_seconds,
            )
            subset_loader = DataLoader(subset_ds, batch_size=args.batch_size, collate_fn=make_collate_fn(feature_extractor))
            subset_metrics = evaluate(model, subset_loader, checkpoint.config.label_names, device)
            per_dataset[dataset_source] = subset_metrics.to_dict()
            print(f"\n  [{dataset_source}, n={subset_metrics.n_samples}] accuracy={subset_metrics.accuracy:.4f} "
                  f"macro_f1={subset_metrics.macro_f1:.4f}")

        existing_metrics = dict(model_version.metrics or {})
        independent_evals = list(existing_metrics.get("independent_evaluations", []))
        independent_evals.append(
            {
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "manifest": str(args.manifest),
                "split": args.split,
                "metrics": metrics.to_dict(),
                "metrics_by_dataset_source": per_dataset,
            }
        )
        existing_metrics["independent_evaluations"] = independent_evals
        model_version.metrics = existing_metrics

        run = await EvaluationRunRepository(session).create(
            evaluation_type="emotion",
            dataset_version=f"{Path(args.manifest).name}:{args.split}",
            model_version=f"{model_version.component}:{model_version.version_tag}",
            configuration={
                "manifest": str(args.manifest), "split": args.split,
                "model_version_id": str(model_version.id), "label_names": checkpoint.config.label_names,
            },
            sample_count=metrics.n_samples,
            status="completed",
            metrics=metrics.to_dict(),
            errors=[],
            notes="Emotion Model v2 independent re-evaluation - reuses the manifest's speaker-independent split unchanged.",
        )
        await session.commit()
        print(f"\nAppended this evaluation to model_versions.metrics.independent_evaluations for {model_version.id}.")
        print(f"Persisted EvaluationRun {run.id} (status=completed).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-version-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument(
        "--device", default="cpu", choices=["auto", "cpu", "mps", "cuda"],
        help="Defaults to cpu for the same reason train_emotion_model_v2.py does - see its --device help.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-audio-seconds", type=float, default=10.0)
    parser.add_argument("--waveform-cache-dir", default="/tmp/voxmind_emotion_v2_waveform_cache")
    args = parser.parse_args()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
