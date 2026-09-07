#!/usr/bin/env python3
"""Trains VoxMind's emotion classifier on a prepared manifest (see
ml/datasets/prepare_emotion_dataset.py) and registers the result as a new
`model_versions` row - offline ML workflow, not run inside the FastAPI app.

This script requires the `voxmind` package to be installed (it is, in
apps/api's venv - run this with that venv's Python) since it reuses the
exact same acoustic feature extractor, Wav2Vec2 provider, preprocessing
function, and classifier architecture that the API uses at inference time.
Reusing Phase 2's `preprocess_audio` here (not just at serve time) matters:
training audio must go through the same mono/16kHz normalization inference
audio does, or the feature distributions the classifier learns on won't
match what it sees in production.

Usage:
    python -m ml.training.train_emotion_model \\
        --manifest ml/datasets/manifests/ravdess_manifest.jsonl \\
        --version-tag ravdess-v1 \\
        --activate

If the manifest doesn't exist, this fails immediately with an actionable
message rather than fabricating one.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.datasets.schema import DatasetSplit, EmotionSample, label_distribution, load_manifest
from ml.datasets.splitting import assert_no_speaker_leakage
from ml.evaluation.metrics import compute_classification_metrics
from ml.training.config import TrainingConfig
from voxmind.core.config import get_settings
from voxmind.services.emotion.acoustic_features import LibrosaAcousticFeatureExtractor
from voxmind.services.emotion.interfaces import ACOUSTIC_VECTOR_DIM
from voxmind.services.emotion.model import (
    EmotionCheckpoint,
    EmotionClassifierConfig,
    EmotionClassifierNet,
    FeatureNormalizer,
)
from voxmind.services.emotion.wav2vec_provider import HuggingFaceWav2Vec2Provider
from voxmind.services.speech.preprocessing import preprocess_audio
from voxmind.services.storage.factory import build_storage_backend

logger = structlog.get_logger(__name__)


def set_seed(seed: int) -> None:
    """Deterministic on CPU for the operations this script uses. Remaining
    source of nondeterminism if this is ever run on a GPU: cuDNN's default
    convolution algorithm selection is nondeterministic unless
    `torch.backends.cudnn.deterministic = True` is also set (not enabled by
    default here since it can noticeably slow down GPU training, and this
    script targets CPU/small-model training)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


async def build_feature_vector(
    sample: EmotionSample,
    *,
    settings,
    extractor: LibrosaAcousticFeatureExtractor,
    embedder: HuggingFaceWav2Vec2Provider,
) -> list[float]:
    raw_bytes = Path(sample.audio_path).read_bytes()
    _metadata, wav_bytes = await preprocess_audio(raw_bytes, settings=settings)
    features = await extractor.extract(wav_bytes, sample_rate=16000)
    embedding = await embedder.embed(wav_bytes, sample_rate=16000)
    return features.to_vector() + embedding.vector


async def extract_all_features(
    samples: list[EmotionSample], *, settings
) -> dict[str, list[float]]:
    extractor = LibrosaAcousticFeatureExtractor()
    embedder = HuggingFaceWav2Vec2Provider(settings)
    vectors: dict[str, list[float]] = {}
    for i, sample in enumerate(samples):
        vectors[sample.sample_id] = await build_feature_vector(
            sample, settings=settings, extractor=extractor, embedder=embedder
        )
        if (i + 1) % 25 == 0 or i == len(samples) - 1:
            logger.info("feature_extraction_progress", completed=i + 1, total=len(samples))
    return vectors


def subset(samples: list[EmotionSample], split: DatasetSplit) -> list[EmotionSample]:
    return [s for s in samples if s.split == split]


def compute_class_weights(labels: list[str], label_names: list[str]) -> torch.Tensor:
    counts = {label: 0 for label in label_names}
    for label in labels:
        counts[label] += 1
    n_samples = len(labels)
    n_classes = len(label_names)
    weights = [
        n_samples / (n_classes * counts[label]) if counts[label] > 0 else 0.0 for label in label_names
    ]
    return torch.tensor(weights, dtype=torch.float32)


async def run_training(args: argparse.Namespace) -> None:
    config = TrainingConfig(
        seed=args.seed,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        early_stopping_patience=args.early_stopping_patience,
        use_class_weights=not args.no_class_weights,
    )
    set_seed(config.seed)
    settings = get_settings()

    samples = load_manifest(args.manifest)
    if any(s.split is None for s in samples):
        print("error: manifest has unsplit samples - run prepare_emotion_dataset.py first.", file=sys.stderr)
        sys.exit(1)
    assert_no_speaker_leakage(samples)

    train_samples = subset(samples, DatasetSplit.TRAIN)
    val_samples = subset(samples, DatasetSplit.VALIDATION)
    test_samples = subset(samples, DatasetSplit.TEST)
    if not train_samples or not val_samples or not test_samples:
        print("error: manifest is missing one of train/validation/test splits.", file=sys.stderr)
        sys.exit(1)

    label_names = sorted({s.label for s in train_samples})
    print(f"Labels (from training split): {label_names}")
    print(f"Train: {len(train_samples)} {label_distribution(train_samples)}")
    print(f"Validation: {len(val_samples)} {label_distribution(val_samples)}")
    print(f"Test: {len(test_samples)} {label_distribution(test_samples)}")

    print("Extracting acoustic features + Wav2Vec2 embeddings for all splits...")
    all_vectors = await extract_all_features(samples, settings=settings)

    train_vectors = [all_vectors[s.sample_id] for s in train_samples]
    normalizer = FeatureNormalizer.fit(train_vectors)

    def to_tensor_dataset(split_samples: list[EmotionSample]) -> TensorDataset:
        features = torch.stack([normalizer.transform(all_vectors[s.sample_id]) for s in split_samples])
        targets = torch.tensor([label_names.index(s.label) for s in split_samples], dtype=torch.long)
        return TensorDataset(features, targets)

    train_ds = to_tensor_dataset(train_samples)
    val_ds = to_tensor_dataset(val_samples)
    test_ds = to_tensor_dataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    model_config = EmotionClassifierConfig(
        acoustic_dim=ACOUSTIC_VECTOR_DIM,
        embedding_dim=768,
        label_names=label_names,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )
    model = EmotionClassifierNet(model_config)

    class_weights = (
        compute_class_weights([s.label for s in train_samples], label_names)
        if config.use_class_weights
        else None
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_val_macro_f1 = -1.0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                logits = model(batch_features)
                preds = logits.argmax(dim=-1)
                val_true.extend(label_names[i] for i in batch_targets.tolist())
                val_pred.extend(label_names[i] for i in preds.tolist())
        val_metrics = compute_classification_metrics(val_true, val_pred, label_names)
        print(f"epoch {epoch + 1}/{config.epochs}: val_macro_f1={val_metrics.macro_f1:.4f}")

        if val_metrics.macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_metrics.macro_f1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {config.early_stopping_patience} epochs).")
                break

    if best_state_dict is None:
        print("error: training produced no valid checkpoint.", file=sys.stderr)
        sys.exit(1)
    model.load_state_dict(best_state_dict)

    model.eval()
    test_true, test_pred = [], []
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            logits = model(batch_features)
            preds = logits.argmax(dim=-1)
            test_true.extend(label_names[i] for i in batch_targets.tolist())
            test_pred.extend(label_names[i] for i in preds.tolist())
    test_metrics = compute_classification_metrics(test_true, test_pred, label_names)

    print("\n=== Final TEST set evaluation (held out, never used for training/model selection) ===")
    print(f"accuracy={test_metrics.accuracy:.4f} macro_f1={test_metrics.macro_f1:.4f} "
          f"weighted_f1={test_metrics.weighted_f1:.4f} balanced_accuracy={test_metrics.balanced_accuracy:.4f}")

    checkpoint = EmotionCheckpoint(config=model_config, normalizer=normalizer, state_dict=best_state_dict)
    storage_key = f"{settings.EMOTION_MODEL_ARTIFACT_ROOT}/{args.version_tag}/classifier.pt"
    storage = build_storage_backend(settings)
    await storage.upload(storage_key, checkpoint.to_bytes(), content_type="application/octet-stream")
    print(f"Saved checkpoint to storage key: {storage_key}")

    mlflow_run_id = None
    try:
        import mlflow

        with mlflow.start_run(run_name=args.version_tag) as run:
            mlflow.log_params(config.to_dict())
            mlflow.log_metric("val_best_macro_f1", best_val_macro_f1)
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in test_metrics.to_dict().items() if isinstance(v, (int, float))}
            )
            mlflow_run_id = run.info.run_id
    except ImportError:
        print("mlflow not installed - skipping experiment tracking (pip install '.[ml-tracking]').")

    from voxmind.db.session import AsyncSessionLocal
    from voxmind.repositories.model_version_repository import ModelVersionRepository

    async with AsyncSessionLocal() as session:
        repo = ModelVersionRepository(session)
        version = await repo.create(
            component="emotion_classifier",
            version_tag=args.version_tag,
            task="emotion_classification",
            base_model=settings.WAV2VEC2_MODEL,
            label_mapping={str(i): label for i, label in enumerate(label_names)},
            training_config=config.to_dict(),
            dataset_info={
                "source": train_samples[0].dataset_source,
                "manifest": str(args.manifest),
                "n_train": len(train_samples),
                "n_validation": len(val_samples),
                "n_test": len(test_samples),
                "train_label_distribution": label_distribution(train_samples),
            },
            artifact_storage_key=storage_key,
            trained=True,
            metrics={"validation": {"best_macro_f1": best_val_macro_f1}, "test": test_metrics.to_dict()},
            mlflow_run_id=mlflow_run_id,
            is_active=False,
        )
        if args.activate:
            await repo.activate(version.id)
        await session.commit()
        print(f"\nRegistered model_versions row: {version.id} (version_tag={args.version_tag})")
        if args.activate:
            print("Activated as the default emotion_classifier model.")
        else:
            print("Not activated - the previous active model (if any) is still in production. "
                  f"Activate with: UPDATE via ModelVersionRepository.activate({version.id}) "
                  "or re-run with --activate.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--version-tag", required=True)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
