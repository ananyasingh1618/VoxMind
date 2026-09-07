#!/usr/bin/env python3
"""Independently re-evaluates an already-registered emotion model version
against a manifest split - separate from (and reusing the same metrics
module as) the training script's internal test evaluation, so a model can
be audited later against a new/updated test set without retraining.

Usage:
    python -m ml.evaluation.evaluate_emotion_model \\
        --model-version-id <uuid> \\
        --manifest ml/datasets/manifests/ravdess_manifest.jsonl \\
        --split test
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ml.datasets.schema import DatasetSplit, load_manifest
from ml.evaluation.metrics import compute_classification_metrics
from voxmind.core.config import get_settings
from voxmind.services.emotion.acoustic_features import LibrosaAcousticFeatureExtractor
from voxmind.services.emotion.classifier_provider import TorchEmotionClassifier
from voxmind.services.emotion.model import EmotionCheckpoint
from voxmind.services.emotion.wav2vec_provider import HuggingFaceWav2Vec2Provider
from voxmind.services.speech.preprocessing import preprocess_audio
from voxmind.services.storage.factory import build_storage_backend


async def run_evaluation(args: argparse.Namespace) -> None:
    from voxmind.db.session import AsyncSessionLocal
    from voxmind.repositories.model_version_repository import ModelVersionRepository

    settings = get_settings()

    from voxmind.repositories.evaluation_run_repository import EvaluationRunRepository

    async with AsyncSessionLocal() as session:
        repo = ModelVersionRepository(session)
        model_version = await repo.get_by_id(uuid.UUID(args.model_version_id))
        if model_version is None:
            print(f"error: no model_versions row with id {args.model_version_id}", file=sys.stderr)
            sys.exit(1)
        if not model_version.artifact_storage_key:
            print("error: this model version has no artifact_storage_key (was it ever trained?).", file=sys.stderr)
            sys.exit(1)

        storage = build_storage_backend(settings)
        checkpoint_bytes = await storage.download(model_version.artifact_storage_key)
        checkpoint = EmotionCheckpoint.from_bytes(checkpoint_bytes)
        classifier = TorchEmotionClassifier.from_checkpoint(
            checkpoint, model_version_id=str(model_version.id), trained=True
        )

        samples = load_manifest(args.manifest)
        target_split = DatasetSplit(args.split)
        eval_samples = [s for s in samples if s.split == target_split]
        if not eval_samples:
            print(f"error: manifest has no samples in split {args.split!r}.", file=sys.stderr)
            sys.exit(1)

        extractor = LibrosaAcousticFeatureExtractor()
        embedder = HuggingFaceWav2Vec2Provider(settings)

        y_true: list[str] = []
        y_pred: list[str] = []
        for i, sample in enumerate(eval_samples):
            raw_bytes = Path(sample.audio_path).read_bytes()
            _metadata, wav_bytes = await preprocess_audio(raw_bytes, settings=settings)
            features = await extractor.extract(wav_bytes, sample_rate=16000)
            embedding = await embedder.embed(wav_bytes, sample_rate=16000)
            prediction = await classifier.predict(features, embedding)
            y_true.append(sample.label)
            y_pred.append(prediction.label)
            if (i + 1) % 25 == 0 or i == len(eval_samples) - 1:
                print(f"evaluated {i + 1}/{len(eval_samples)}")

        metrics = compute_classification_metrics(y_true, y_pred, checkpoint.config.label_names)

        print(f"\n=== Independent evaluation: model {model_version.id} ({model_version.version_tag}) ===")
        print(f"split={args.split} n={metrics.n_samples}")
        print(f"accuracy={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f} "
              f"weighted_f1={metrics.weighted_f1:.4f} balanced_accuracy={metrics.balanced_accuracy:.4f}")
        print("per-class:")
        for label, m in metrics.per_class.items():
            print(f"  {label}: precision={m.precision:.3f} recall={m.recall:.3f} f1={m.f1:.3f} support={m.support}")
        print(f"confusion_matrix (rows=true, cols=pred, order={metrics.label_order}):")
        for row in metrics.confusion_matrix:
            print(f"  {row}")

        existing_metrics = dict(model_version.metrics or {})
        independent_evals = list(existing_metrics.get("independent_evaluations", []))
        independent_evals.append(
            {
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "manifest": str(args.manifest),
                "split": args.split,
                "metrics": metrics.to_dict(),
            }
        )
        existing_metrics["independent_evaluations"] = independent_evals
        model_version.metrics = existing_metrics

        # Phase 7: also record this run in the shared EvaluationRun ledger
        # (additive only - the append-only model_versions.metrics update
        # above, the speaker-independent split, and the held-out test set
        # are all unchanged from Phase 3).
        run = await EvaluationRunRepository(session).create(
            evaluation_type="emotion",
            dataset_version=f"{Path(args.manifest).name}:{args.split}",
            model_version=f"{model_version.component}:{model_version.version_tag}",
            configuration={
                "manifest": str(args.manifest),
                "split": args.split,
                "model_version_id": str(model_version.id),
                "label_names": checkpoint.config.label_names,
            },
            sample_count=metrics.n_samples,
            status="completed",
            metrics=metrics.to_dict(),
            errors=[],
            notes="Reuses the Phase 3 speaker-independent held-out test split - not re-derived or altered.",
        )
        await session.commit()
        print(f"\nAppended this evaluation to model_versions.metrics.independent_evaluations for {model_version.id}.")
        print(f"Persisted EvaluationRun {run.id} (status=completed).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-version-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    args = parser.parse_args()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
