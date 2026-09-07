#!/usr/bin/env python3
"""Trains the emotion2vec+-based production candidate (docs/emotion.md):
Stage A only by default - the emotion2vec+ encoder stays entirely frozen
(`--freeze-encoder`, on by default; there is no encoder-fine-tuning path
in this script at all, since Stage B was not found to be necessary - see
docs/emotion.md's "Results" section for why) and only a small
`Emotion2VecHead` (LayerNorm -> Linear -> GELU -> Dropout -> Linear(6)) is
trained, on cached, frozen 768-dim embeddings.

Because the encoder never receives gradients, this script never needs the
memory-conservative machinery `train_emotion_model_v2.py` required (no
micro-batching, no gradient accumulation, no gradient checkpointing): the
entire cached embedding matrix for every split fits trivially in memory
(under 10MB total for ~2,800 clips at 768 floats each), so training is a
plain, fast, in-memory loop - a deliberately much simpler script than v2's,
reflecting a genuinely simpler (and, per docs/emotion.md, more
practically sound) architecture choice, not a shortcut.

Usage:
    python -m ml.training.train_emotion2vec \\
        --manifest ml/datasets/manifests/emotion2vec_manifest.jsonl \\
        --ravdess-manifest ml/datasets/manifests/ravdess_manifest.jsonl \\
        --version-tag emotion2vec-tess-emodb-v1 --seed 42
    (add --activate only after reviewing the evaluation)
"""
from __future__ import annotations

import argparse
import asyncio
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import __version__ as transformers_version

from ml.datasets.emotion2vec_labels import RAVDESS_TO_V2, V2_LABELS
from ml.datasets.schema import DatasetSplit, EmotionSample, label_distribution, load_manifest
from ml.datasets.splitting import assert_no_speaker_leakage
from ml.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from ml.training.train_emotion_model import compute_class_weights, set_seed
from voxmind.core.config import get_settings
from voxmind.services.emotion.emotion2vec.encoder import EMBEDDING_DIM, Emotion2VecEncoder
from voxmind.services.emotion.emotion2vec.model import Emotion2VecCheckpoint, Emotion2VecHead, Emotion2VecHeadConfig
from voxmind.services.speech.preprocessing import preprocess_audio
from voxmind.services.storage.factory import build_storage_backend

try:
    import funasr as _funasr  # noqa: F401
except ImportError:
    print("error: the 'funasr' package is required (pip install funasr) - see docs/emotion.md.", file=sys.stderr)
    sys.exit(1)


async def _embed_one(encoder: Emotion2VecEncoder, settings, audio_path: str) -> np.ndarray:
    raw_bytes = Path(audio_path).read_bytes()
    _metadata, wav_bytes = await preprocess_audio(raw_bytes, settings=settings)
    embedding = await encoder.embed(wav_bytes, sample_rate=16000)
    return np.asarray(embedding.vector, dtype=np.float32)


async def cache_embeddings(
    samples: list[EmotionSample], *, encoder: Emotion2VecEncoder, settings, cache_dir: Path
) -> None:
    """Real Phase-2 preprocessing (`preprocess_audio` - same ffmpeg mono/
    16kHz normalization every other emotion model in this project uses)
    followed by a real frozen emotion2vec+ forward pass, cached to disk as
    one .npy per sample - idempotent, matching train_emotion_model_v2.py's
    established pattern."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    to_process = [s for s in samples if not (cache_dir / f"{s.sample_id}.npy").exists()]
    if len(to_process) < len(samples):
        print(f"  {len(samples) - len(to_process)}/{len(samples)} already cached from a previous run.")
    for i, sample in enumerate(to_process):
        vector = await _embed_one(encoder, settings, sample.audio_path)
        np.save(cache_dir / f"{sample.sample_id}.npy", vector)
        if (i + 1) % 100 == 0 or i == len(to_process) - 1:
            print(f"  embedded {i + 1}/{len(to_process)}")


def load_cached(samples: list[EmotionSample], *, cache_dir: Path, label_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = np.stack([np.load(cache_dir / f"{s.sample_id}.npy") for s in samples])
    labels = torch.tensor([label_names.index(s.label) for s in samples], dtype=torch.long)
    return torch.tensor(vectors, dtype=torch.float32), labels


def evaluate_head(model: Emotion2VecHead, embeddings: torch.Tensor, labels: torch.Tensor, label_names: list[str]) -> ClassificationMetrics:
    model.eval()
    with torch.no_grad():
        logits = model(embeddings)
        preds = logits.argmax(dim=-1)
    y_true = [label_names[i] for i in labels.tolist()]
    y_pred = [label_names[i] for i in preds.tolist()]
    return compute_classification_metrics(y_true, y_pred, label_names)


async def load_ravdess_robustness_set(ravdess_manifest_path: str, *, encoder: Emotion2VecEncoder, settings, cache_dir: Path):
    """RAVDESS is used ONLY as an external robustness benchmark - reuses
    the existing, frozen `ravdess-v1` speaker-independent TEST split
    unchanged (docs/emotion.md), never trained on here. Six-class-mapped
    via the same RAVDESS_TO_V2 mapping v2 already established (calm/
    surprised excluded, never remapped)."""
    samples = load_manifest(ravdess_manifest_path)
    test_samples = [s for s in samples if s.split == DatasetSplit.TEST]
    mapped = []
    for s in test_samples:
        v2_label = RAVDESS_TO_V2.get(s.label)
        if v2_label is not None:
            from dataclasses import replace

            mapped.append(replace(s, label=v2_label, sample_id=f"ravdess-robustness-{s.sample_id}"))
    print(f"RAVDESS robustness set: {len(mapped)} six-class-mapped clips "
          f"(from {len(test_samples)} raw test clips, calm/surprised excluded)")
    await cache_embeddings(mapped, encoder=encoder, settings=settings, cache_dir=cache_dir)
    return mapped


async def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    settings = get_settings()

    samples = load_manifest(args.manifest)
    if any(s.split is None for s in samples):
        print("error: manifest has unsplit samples - run prepare_emotion2vec_dataset.py first.", file=sys.stderr)
        sys.exit(1)
    assert_no_speaker_leakage(samples)

    train_samples = [s for s in samples if s.split == DatasetSplit.TRAIN]
    val_samples = [s for s in samples if s.split == DatasetSplit.VALIDATION]
    test_samples = [s for s in samples if s.split == DatasetSplit.TEST]
    label_names = V2_LABELS
    print(f"Labels: {label_names}")
    print(f"Train: {len(train_samples)} {label_distribution(train_samples)}")
    print(f"Validation: {len(val_samples)} {label_distribution(val_samples)}")
    print(f"Test: {len(test_samples)} {label_distribution(test_samples)}")

    encoder = Emotion2VecEncoder(settings)
    cache_dir = Path(args.embedding_cache_dir)
    print(f"\nExtracting frozen emotion2vec+ embeddings (cached to {cache_dir}) ...")
    await cache_embeddings(samples, encoder=encoder, settings=settings, cache_dir=cache_dir)

    train_x, train_y = load_cached(train_samples, cache_dir=cache_dir, label_names=label_names)
    val_x, val_y = load_cached(val_samples, cache_dir=cache_dir, label_names=label_names)
    test_x, test_y = load_cached(test_samples, cache_dir=cache_dir, label_names=label_names)

    config = Emotion2VecHeadConfig(embedding_dim=EMBEDDING_DIM, label_names=label_names, hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = Emotion2VecHead(config)

    class_weights = compute_class_weights([s.label for s in train_samples], label_names)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)

    best_val_macro_f1 = -1.0
    best_state: dict | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    training_started = time.monotonic()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        val_metrics = evaluate_head(model, val_x, val_y, label_names)
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={epoch_loss / max(n_batches, 1):.4f} val_macro_f1={val_metrics.macro_f1:.4f}")

        if val_metrics.macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_metrics.macro_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {args.early_stopping_patience} epochs).")
                break

    training_duration_seconds = time.monotonic() - training_started
    if best_state is None:
        print("error: training produced no valid checkpoint.", file=sys.stderr)
        sys.exit(1)
    model.load_state_dict(best_state)

    test_metrics = evaluate_head(model, test_x, test_y, label_names)
    print("\n=== Final TEST set evaluation (combined TESS+EMO-DB, held out) ===")
    print(f"accuracy={test_metrics.accuracy:.4f} macro_f1={test_metrics.macro_f1:.4f} "
          f"weighted_f1={test_metrics.weighted_f1:.4f} balanced_accuracy={test_metrics.balanced_accuracy:.4f}")

    per_dataset_metrics: dict[str, dict] = {}
    for dataset_source in sorted({s.dataset_source for s in test_samples}):
        subset = [s for s in test_samples if s.dataset_source == dataset_source]
        sub_x, sub_y = load_cached(subset, cache_dir=cache_dir, label_names=label_names)
        sub_metrics = evaluate_head(model, sub_x, sub_y, label_names)
        per_dataset_metrics[dataset_source] = sub_metrics.to_dict()
        print(f"  [{dataset_source} test subset, n={sub_metrics.n_samples}] accuracy={sub_metrics.accuracy:.4f} macro_f1={sub_metrics.macro_f1:.4f}")

    ravdess_metrics_dict = None
    if args.ravdess_manifest:
        ravdess_cache_dir = Path(args.ravdess_embedding_cache_dir)
        ravdess_samples = await load_ravdess_robustness_set(args.ravdess_manifest, encoder=encoder, settings=settings, cache_dir=ravdess_cache_dir)
        rav_x, rav_y = load_cached(ravdess_samples, cache_dir=ravdess_cache_dir, label_names=label_names)
        ravdess_metrics = evaluate_head(model, rav_x, rav_y, label_names)
        ravdess_metrics_dict = ravdess_metrics.to_dict()
        print("\n=== RAVDESS external robustness benchmark (never trained on) ===")
        print(f"accuracy={ravdess_metrics.accuracy:.4f} macro_f1={ravdess_metrics.macro_f1:.4f} "
              f"weighted_f1={ravdess_metrics.weighted_f1:.4f} balanced_accuracy={ravdess_metrics.balanced_accuracy:.4f}")

    model.eval()
    checkpoint = Emotion2VecCheckpoint(
        config=config,
        state_dict=model.state_dict(),
        provenance={
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "learning_rate": args.learning_rate,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "batch_size": args.batch_size,
            "encoder_frozen": True,
            "stage": "A",
            "training_duration_seconds": training_duration_seconds,
            "hardware": platform.platform(),
            "library_versions": {"torch": torch.__version__, "transformers": transformers_version, "funasr": _funasr.__version__},
            "encoder_model": settings.EMOTION2VEC_MODEL,
        },
    )
    storage_key = f"{settings.EMOTION_MODEL_ARTIFACT_ROOT}/{args.version_tag}/classifier.pt"
    storage = build_storage_backend(settings)
    checkpoint_bytes = checkpoint.to_bytes()
    await storage.upload(storage_key, checkpoint_bytes, content_type="application/octet-stream")
    print(f"\nSaved checkpoint to storage key: {storage_key} ({len(checkpoint_bytes) / 1e6:.2f} MB)")

    from voxmind.db.session import AsyncSessionLocal
    from voxmind.repositories.model_version_repository import ModelVersionRepository

    async with AsyncSessionLocal() as session:
        repo = ModelVersionRepository(session)
        version = await repo.create(
            component="emotion_classifier",
            version_tag=args.version_tag,
            task="emotion_classification",
            base_model=settings.EMOTION2VEC_MODEL,
            label_mapping={str(i): label for i, label in enumerate(label_names)},
            training_config={"architecture": "emotion2vec-plus-base", **checkpoint.provenance},
            dataset_info={
                "manifest": str(args.manifest),
                "n_train": len(train_samples),
                "n_validation": len(val_samples),
                "n_test": len(test_samples),
                "train_label_distribution": label_distribution(train_samples),
                "train_dataset_sources": sorted({s.dataset_source for s in train_samples}),
            },
            artifact_storage_key=storage_key,
            trained=True,
            metrics={
                "validation": {"best_macro_f1": best_val_macro_f1},
                "test": test_metrics.to_dict(),
                "test_by_dataset_source": per_dataset_metrics,
                "ravdess_robustness": ravdess_metrics_dict,
            },
            is_active=False,
        )
        if args.activate:
            await repo.activate(version.id)
        await session.commit()
        print(f"\nRegistered model_versions row: {version.id} (version_tag={args.version_tag})")
        print("Activated." if args.activate else "Not activated - review the evaluation above before activating.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ravdess-manifest", default=None, help="Path to the existing frozen ravdess_manifest.jsonl for the robustness benchmark.")
    parser.add_argument("--version-tag", required=True)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--freeze-encoder", action="store_true", default=True, help="Always true in this script - Stage A only, see module docstring.")
    parser.add_argument("--embedding-cache-dir", default="/tmp/voxmind_emotion2vec_embedding_cache")
    parser.add_argument("--ravdess-embedding-cache-dir", default="/tmp/voxmind_emotion2vec_ravdess_embedding_cache")
    args = parser.parse_args()
    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
