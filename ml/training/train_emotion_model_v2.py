#!/usr/bin/env python3
"""Trains Emotion Model v2 (docs/emotion.md) on the unified RAVDESS+CREMA-D
six-class manifest (ml/datasets/prepare_emotion_dataset_v2.py) and
registers the result as a NEW `model_versions` row - never touching or
overwriting the frozen `ravdess-v1` row or checkpoint.

Genuinely different from v1's training script in one fundamental way: v1
precomputes frozen Wav2Vec2 embeddings once and trains a small MLP on top
(train_emotion_model.py); v2 *fine-tunes* Wav2Vec2 itself, so every epoch
must re-run the backbone's forward pass over raw waveforms - there is no
"precompute once" shortcut once the backbone's weights are allowed to
change. This script otherwise deliberately mirrors v1's script structure
(CLI shape, early stopping on validation macro-F1, best-checkpoint
selection, final held-out test evaluation, MLflow logging, ModelVersion
registration) so the two are easy to compare and audit side by side.

Two-stage fine-tuning (docs/emotion.md, "Training strategy"):
  Stage 1: backbone frozen entirely; only attention-pooling + head train.
  Stage 2: the top `--unfreeze-layers` transformer encoder layers (closest
    to the head) are unfrozen at a much smaller learning rate; pooling +
    head continue training at a smaller (not the stage-1) rate to avoid
    destroying what stage 1 already learned while still letting them
    adapt to the now-changing backbone representation.
A single "best checkpoint so far" (by validation macro-F1) is tracked
across BOTH stages combined - exactly one final model is selected, not one
per stage.

Usage:
    python -m ml.training.train_emotion_model_v2 \\
        --manifest ml/datasets/manifests/emotion_v2_manifest.jsonl \\
        --version-tag emotion-v2 --seed 42
    (add --activate only after reviewing the evaluation - see
    docs/emotion.md's "Acceptance criteria" / promotion decision)

Memory-conservative defaults (docs/emotion.md, "Training safety"): a real
full-scale run on this project's actual hardware (an 8GB-unified-memory
Apple M2, no CUDA) initially drove the machine into severe, sustained
swap thrashing under MPS - confirmed via `sample`/`vm_stat` (physical
footprint reached ~7.8GB, ~98% of total RAM). This script's defaults were
redesigned specifically for that constraint, not loosened back to
"whatever's fastest" once a smoke/timing test looked fine on a small
subset: `--device cpu` (not mps/auto), a tiny `--batch-size` (1) combined
with `--grad-accum-steps` (8, giving the same effective batch size a
larger true batch would) so no single forward/backward pass ever holds
more than one clip's activations, `--gradient-checkpointing` enabled for
stage 2 (trades recompute for memory on the newly-unfrozen layers), and
`--unfreeze-layers` reduced to 1. None of this changes the architecture
(model.py is untouched) or the dataset - only how much of each is held in
memory in flight at once. Real, per-batch RSS is logged during training
(see current_rss_mb()) specifically so a repeat of this problem would be
visible in the log itself next time, not only discoverable via external
process inspection after the fact.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import io
import platform
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import structlog
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor
from transformers import __version__ as transformers_version

from ml.datasets.augmentation import AugmentationConfig, apply_augmentation
from ml.datasets.schema import DatasetSplit, EmotionSample, label_distribution, load_manifest
from ml.datasets.splitting import assert_no_speaker_leakage
from ml.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from ml.training.focal_loss import FocalLossWithLabelSmoothing
from ml.training.train_emotion_model import compute_class_weights, set_seed
from voxmind.core.config import get_settings
from voxmind.services.emotion.v2.model import (
    EmotionCheckpointV2,
    EmotionClassifierV2Config,
    Wav2VecAttentionEmotionNet,
)
from voxmind.services.speech.preprocessing import preprocess_audio
from voxmind.services.storage.factory import build_storage_backend

logger = structlog.get_logger(__name__)

MAX_AUDIO_SECONDS_DEFAULT = 10.0  # generous cap - RAVDESS/CREMA-D clips are all short utterances


class RawWaveformDataset(Dataset):
    """Returns raw (float32, 16kHz, mono) waveforms - not precomputed
    features, since v2 fine-tunes the backbone every epoch (see module
    docstring). Reuses `preprocess_audio` (Phase 2) so training audio goes
    through the exact same ffmpeg mono/16kHz normalization inference audio
    does - the same train/serve-skew concern v1's docstring already flags,
    unchanged here.

    A real bug the mandatory smoke-training run (docs/emotion.md, "Training
    safety") caught before any real training happened: `preprocess_audio`
    is async (it shells out to ffmpeg via `asyncio.create_subprocess_exec`),
    but `Dataset.__getitem__` must be synchronous - an earlier version of
    this class called `asyncio.run(preprocess_audio(...))` from inside
    `__getitem__`, which crashes with "asyncio.run() cannot be called from
    a running event loop" because the whole script already runs inside
    `asyncio.run(run_training(...))`. Fixed by precomputing every sample's
    preprocessed (mono, 16kHz) waveform ONCE, upfront, in an async context
    (`preprocess_and_cache_waveforms` below) - this dataset only ever does
    synchronous, on-disk-cached array loading/augmentation, and as a real
    side benefit (not just a workaround) avoids re-running ffmpeg on every
    sample on every epoch, mirroring v1's own "extract features once"
    pattern (train_emotion_model.py::extract_all_features) - with one
    further, load-bearing difference from v1: the cache lives on disk, not
    in an in-memory dict. A first real full-scale run of this script
    (~8,500 clips - much larger than v1's 1,440) *did* try the in-memory
    version first and drove this 8GB-unified-memory machine into heavy
    swapping (confirmed via `vm_stat`) - see
    preprocess_and_cache_waveforms's docstring for the full account."""

    def __init__(
        self,
        samples: list[EmotionSample],
        *,
        cache_dir: Path,
        label_names: list[str],
        augment: bool,
        augmentation_config: AugmentationConfig,
        seed: int,
        max_audio_seconds: float,
        sample_rate: int = 16000,
    ) -> None:
        self.samples = samples
        self.cache_dir = cache_dir
        self.label_names = label_names
        self.augment = augment
        self.augmentation_config = augmentation_config
        self.max_audio_seconds = max_audio_seconds
        self.sample_rate = sample_rate
        # One independent, deterministic RNG per dataset index so re-reading
        # the same index within an epoch (shouldn't happen with shuffle,
        # but DataLoader workers could in principle) still gets a
        # reproducible-for-that-index augmentation, not global RNG state
        # shared/raced across workers.
        self._seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, str]:
        sample = self.samples[index]
        # Loaded from the on-disk cache (see preprocess_and_cache_waveforms)
        # rather than kept resident in memory - see that function's
        # docstring for the real out-of-memory this replaced. A single
        # small (<1MB) np.load per __getitem__ call is cheap relative to
        # the Wav2Vec2 forward/backward pass that follows it.
        waveform = np.load(self.cache_dir / f"{sample.sample_id}.npy")
        max_samples = int(self.max_audio_seconds * self.sample_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        if self.augment:
            rng = random.Random(self._seed * 1_000_003 + index)
            waveform = apply_augmentation(waveform, rng=rng, config=self.augmentation_config)

        label_idx = self.label_names.index(sample.label)
        # sample_id is returned alongside the tensor data (not used by the
        # model at all) specifically so a slow/stalled batch can be traced
        # back to the exact file responsible - see train_one_stage's
        # slow-batch alarm, added after a real stall during a full-scale
        # run that took hours to even notice, let alone diagnose, without this.
        return waveform.astype(np.float32), label_idx, sample.sample_id


async def preprocess_and_cache_waveforms(samples: list[EmotionSample], *, settings, cache_dir: Path) -> None:
    """Runs Phase 2's real ffmpeg-based `preprocess_audio` once per sample
    (mono, 16kHz - see its own module docstring) and writes the resulting
    waveform to `cache_dir/{sample_id}.npy`. Called once, before training
    starts, from the top-level async `run_training`.

    A real out-of-memory problem the first real full-scale training attempt
    hit (docs/emotion.md): an earlier version of this function returned an
    in-memory `dict[str, np.ndarray]` covering the *entire* dataset (all
    train+validation+test clips, ~8,500 for the combined RAVDESS+CREMA-D
    manifest - a very different scale from v1's 1,440-clip RAVDESS-only
    training). Observed directly in this environment: the training
    process's real memory footprint grew to ~5.4GB and drove the (8GB
    unified-memory Apple M2) system into heavy swapping, confirmed via
    `vm_stat`'s free-page count dropping to a few thousand pages. Caching
    to disk instead keeps only one batch's worth of raw waveforms resident
    in memory at any time - RawWaveformDataset.__getitem__ loads on demand
    - at the cost of a small amount of disk I/O per __getitem__ call,
    which is negligible next to the Wav2Vec2 forward/backward pass that
    follows it. Already-cached files are skipped (idempotent - a killed
    and restarted run doesn't redo this step from scratch)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    to_process = [s for s in samples if not (cache_dir / f"{s.sample_id}.npy").exists()]
    if len(to_process) < len(samples):
        print(f"  {len(samples) - len(to_process)}/{len(samples)} already cached from a previous run.")
    for i, sample in enumerate(to_process):
        raw_bytes = Path(sample.audio_path).read_bytes()
        _metadata, wav_bytes = await preprocess_audio(raw_bytes, settings=settings)
        waveform, _sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        np.save(cache_dir / f"{sample.sample_id}.npy", waveform.astype(np.float32))
        if (i + 1) % 100 == 0 or i == len(to_process) - 1:
            print(f"  preprocessed {i + 1}/{len(to_process)}")


def make_collate_fn(feature_extractor: Wav2Vec2FeatureExtractor):
    def collate(batch: list[tuple[np.ndarray, int, str]]):
        waveforms = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        sample_ids = [item[2] for item in batch]
        durations = [len(w) / 16000 for w in waveforms]
        inputs = feature_extractor(
            waveforms, sampling_rate=16000, padding=True, return_tensors="pt"
        )
        return inputs["input_values"], inputs.get("attention_mask"), labels, sample_ids, durations

    return collate


def subset(samples: list[EmotionSample], split: DatasetSplit) -> list[EmotionSample]:
    return [s for s in samples if s.split == split]


@torch.no_grad()
def evaluate(model: Wav2VecAttentionEmotionNet, loader: DataLoader, label_names: list[str], device: str) -> ClassificationMetrics:
    model.eval()
    y_true: list[str] = []
    y_pred: list[str] = []
    for input_values, attention_mask, labels, _sample_ids, _durations in loader:
        input_values = input_values.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        logits = model(input_values, attention_mask=attention_mask)
        preds = logits.argmax(dim=-1).cpu()
        y_true.extend(label_names[i] for i in labels.tolist())
        y_pred.extend(label_names[i] for i in preds.tolist())
    return compute_classification_metrics(y_true, y_pred, label_names)


def save_resume_checkpoint(
    path: Path, *, stage_name: str, epoch_in_stage: int, model: "Wav2VecAttentionEmotionNet",
    optimizer: torch.optim.Optimizer, best_val_macro_f1: float, best_state: dict | None,
    best_epoch: int, global_epoch: int, epochs_without_improvement: int,
) -> None:
    """Saved after EVERY epoch (both stages) - added after a real full-scale
    run stalled for hours on a single batch and had to be killed with zero
    way to recover the several hours of already-completed epochs
    (docs/emotion.md, "Training safety"). Deliberately a *local temp file*,
    not the final registered artifact - this is resume/crash-recovery
    state, not a trained model worth keeping once training finishes
    successfully (main() removes it then)."""
    torch.save(
        {
            "stage_name": stage_name,
            "epoch_in_stage": epoch_in_stage,
            "global_epoch": global_epoch,
            "backbone_state_dict": model.backbone.state_dict(),
            "pooling_state_dict": model.pooling.state_dict(),
            "head_state_dict": model.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "best_state": best_state,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        },
        path,
    )


def load_resume_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def current_rss_mb() -> float:
    """Real, direct memory self-observation - added after a real full-scale
    training run drove this 8GB-unified-memory machine into severe swap
    thrashing (docs/emotion.md's "Training safety" section documents the
    full account) with *zero visibility in the training log itself* -
    diagnosing it required external tools (`sample`, `vm_stat`) run against
    the process from outside. `ru_maxrss` is macOS-reported in bytes (Linux
    reports kilobytes - this project only targets macOS/CPU-or-MPS
    training, so no platform branch is added for a case that isn't this
    project's real environment)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def release_device_memory(device: str) -> None:
    """Called between batches/epochs specifically because the *first* real
    full-scale training attempt exhibited severe, sustained memory growth
    that plain Python garbage collection alone did not visibly relieve -
    MPS's caching allocator is a known real source of exactly this kind of
    growth under many iterations of *variable-shaped* inputs (this
    project's clips have genuinely different durations, so padded batch
    shapes differ batch-to-batch - not a hypothetical concern). Cheap
    relative to a training step; called unconditionally rather than only
    "when memory looks high," since by the time a Python-level check would
    fire, MPS's own internal buffers may already be fragmented."""
    gc.collect()
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


async def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    settings = get_settings()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    samples = load_manifest(args.manifest)
    if any(s.split is None for s in samples):
        print("error: manifest has unsplit samples - run prepare_emotion_dataset_v2.py first.", file=sys.stderr)
        sys.exit(1)
    assert_no_speaker_leakage(samples)

    if args.restrict_train_to_dataset:
        print(f"Cross-corpus diagnostic mode: restricting TRAIN split to dataset_source={args.restrict_train_to_dataset!r} "
              "(validation/test splits are unrestricted, so cross-corpus generalization is directly observable).")

    train_samples = [
        s for s in subset(samples, DatasetSplit.TRAIN)
        if args.restrict_train_to_dataset is None or s.dataset_source == args.restrict_train_to_dataset
    ]
    val_samples = subset(samples, DatasetSplit.VALIDATION)
    test_samples = subset(samples, DatasetSplit.TEST)
    if not train_samples or not val_samples or not test_samples:
        print("error: manifest/filters produced an empty train/validation/test split.", file=sys.stderr)
        sys.exit(1)

    label_names = sorted({s.label for s in samples})  # fixed six-class vocabulary, not just what's in this train subset
    print(f"Labels: {label_names}")
    print(f"Train: {len(train_samples)} {label_distribution(train_samples)}")
    print(f"Validation: {len(val_samples)} {label_distribution(val_samples)}")
    print(f"Test: {len(test_samples)} {label_distribution(test_samples)}")

    cache_dir = Path(args.waveform_cache_dir)
    print(f"Preprocessing {len(samples)} clips (ffmpeg mono/16kHz normalization, once, cached to {cache_dir}) ...")
    await preprocess_and_cache_waveforms(samples, settings=settings, cache_dir=cache_dir)

    augmentation_config = AugmentationConfig()
    train_ds = RawWaveformDataset(
        train_samples, cache_dir=cache_dir, label_names=label_names, augment=not args.no_augmentation,
        augmentation_config=augmentation_config, seed=args.seed, max_audio_seconds=args.max_audio_seconds,
    )
    val_ds = RawWaveformDataset(
        val_samples, cache_dir=cache_dir, label_names=label_names, augment=False,
        augmentation_config=augmentation_config, seed=args.seed, max_audio_seconds=args.max_audio_seconds,
    )
    test_ds = RawWaveformDataset(
        test_samples, cache_dir=cache_dir, label_names=label_names, augment=False,
        augmentation_config=augmentation_config, seed=args.seed, max_audio_seconds=args.max_audio_seconds,
    )

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(settings.WAV2VEC2_MODEL)
    collate_fn = make_collate_fn(feature_extractor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    backbone_hidden_dim = _probe_hidden_dim(settings.WAV2VEC2_MODEL)
    model_config = EmotionClassifierV2Config(
        base_model=settings.WAV2VEC2_MODEL, backbone_hidden_dim=backbone_hidden_dim, label_names=label_names,
    )
    model = Wav2VecAttentionEmotionNet(model_config)
    # Real, pretrained Wav2Vec2 weights loaded once here (the only point in
    # this script that downloads backbone weights) - everything after this
    # trains starting from them, exactly like v1's frozen-embedding
    # approach starts from the same pretrained checkpoint.
    from transformers import Wav2Vec2Model as HFWav2Vec2Model

    pretrained = HFWav2Vec2Model.from_pretrained(settings.WAV2VEC2_MODEL)
    model.backbone.load_state_dict(pretrained.state_dict())
    del pretrained
    model.to(device)

    # Resume support (docs/emotion.md, "Training safety"): a real full-
    # scale run stalled for hours on a single batch and had to be killed,
    # discarding several hours of already-completed epochs with no way to
    # recover them - this closes that gap. `--resume` loads whatever
    # `save_resume_checkpoint` last wrote (after every completed epoch,
    # both stages) and continues from exactly there; without it, training
    # always starts fresh from the downloaded pretrained backbone above.
    resume_path = Path(args.resume_checkpoint_path)
    resume_data = load_resume_checkpoint(resume_path) if args.resume else None
    if resume_data is not None:
        model.backbone.load_state_dict(resume_data["backbone_state_dict"])
        model.pooling.load_state_dict(resume_data["pooling_state_dict"])
        model.head.load_state_dict(resume_data["head_state_dict"])
        model.to(device)
        print(f"Resumed from {resume_path}: stage={resume_data['stage_name']} "
              f"epoch_in_stage={resume_data['epoch_in_stage']} global_epoch={resume_data['global_epoch']}")

    class_weights = compute_class_weights([s.label for s in train_samples], label_names).to(device)
    criterion = FocalLossWithLabelSmoothing(
        num_classes=len(label_names), gamma=args.focal_gamma, label_smoothing=args.label_smoothing, alpha=class_weights
    )

    if resume_data is not None:
        best_val_macro_f1 = resume_data["best_val_macro_f1"]
        best_state = resume_data["best_state"]
        best_epoch = resume_data["best_epoch"]
        global_epoch = resume_data["global_epoch"]
    else:
        best_val_macro_f1 = -1.0
        best_state = None
        best_epoch = -1
        global_epoch = 0
    training_started = time.monotonic()

    def train_one_stage(
        stage_name: str, epochs: int, optimizer: torch.optim.Optimizer, patience: int, *, grad_accum_steps: int,
        start_epoch_in_stage: int = 0, initial_epochs_without_improvement: int = 0,
    ) -> None:
        nonlocal best_val_macro_f1, best_state, best_epoch, global_epoch
        epochs_without_improvement = initial_epochs_without_improvement
        for epoch_in_stage in range(start_epoch_in_stage, epochs):
            global_epoch += 1
            model.train()
            # Keep the frozen backbone in eval() (disables its internal
            # dropout) whenever none of its parameters currently require
            # grad - correct for stage 1, and for the frozen *lower*
            # layers even during stage 2 (only the unfrozen top layers
            # should see dropout as "training").
            backbone_has_trainable_params = any(p.requires_grad for p in model.backbone.parameters())
            model.backbone.train(mode=backbone_has_trainable_params)

            epoch_loss = 0.0
            n_batches = 0
            optimizer.zero_grad()
            epoch_started = time.monotonic()
            last_checkpoint_time = epoch_started
            last_checkpoint_batch = 0
            for batch_idx, (input_values, attention_mask, labels, sample_ids, durations) in enumerate(train_loader):
                batch_started = time.monotonic()
                input_values = input_values.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                logits = model(input_values, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                # Gradient accumulation (docs/emotion.md, "Training safety"):
                # a tiny micro-batch (as small as 1) keeps any single
                # forward/backward pass's activation memory small - this
                # project's real, demonstrated memory constraint, not a
                # theoretical one - while `grad_accum_steps` micro-batches'
                # gradients are summed before one optimizer step, so the
                # *effective* batch size (and therefore the loss landscape
                # the optimizer actually sees) matches what a larger true
                # batch size would produce. Scaling the loss by
                # 1/grad_accum_steps before backward (not after) is what
                # makes accumulated gradients equivalent to a true-large-
                # batch average rather than a sum.
                (loss / grad_accum_steps).backward()
                epoch_loss += float(loss.item())
                n_batches += 1

                is_accumulation_boundary = (batch_idx + 1) % grad_accum_steps == 0
                is_last_batch = batch_idx == len(train_loader) - 1
                if is_accumulation_boundary or is_last_batch:
                    optimizer.step()
                    optimizer.zero_grad()

                # Slow-batch alarm: fires immediately, not only at the next
                # 50-batch checkpoint below. Added after a real full-scale
                # run stalled on ONE batch for hours (docs/emotion.md,
                # "Training safety") - the process was confirmed still
                # genuinely computing (a real stack sample caught it mid
                # backward-pass, not deadlocked), so the earlier 50-batch-
                # interval logging left a multi-hour blind window before
                # anything looked wrong. This can't previously have been
                # reproduced in isolated profiling; logging the exact
                # sample_id/duration the moment a single batch takes far
                # longer than normal is what actually makes a repeat of
                # this diagnosable in seconds instead of hours.
                batch_duration = time.monotonic() - batch_started
                if batch_duration > 5.0:
                    print(f"  SLOW BATCH WARNING: [{stage_name}] epoch {global_epoch} batch {batch_idx + 1} "
                          f"took {batch_duration:.1f}s (sample_ids={sample_ids}, durations_sec={durations})")

                if (batch_idx + 1) % 50 == 0:
                    release_device_memory(device)
                    # Real wall-clock throughput, not just a batch counter -
                    # added after a real full-scale run took ~21x longer
                    # than a short, isolated timing test predicted, for
                    # reasons that turned out NOT to be reproducible in
                    # isolated profiling (forward/backward/data-loading/
                    # gc.collect() were all individually confirmed fast and
                    # stable - docs/emotion.md's "Training safety" section
                    # has the full account). Whatever caused that
                    # slowdown - almost certainly real-world contention on
                    # a shared, memory-constrained machine over a multi-hour
                    # window, not a bug this project's code can fix outright
                    # - this makes it visible AS IT HAPPENS, in wall-clock
                    # terms, rather than only discoverable hours later.
                    now = time.monotonic()
                    recent_batches = (batch_idx + 1) - last_checkpoint_batch
                    seconds_per_batch = (now - last_checkpoint_time) / max(recent_batches, 1)
                    remaining_batches = len(train_loader) - (batch_idx + 1)
                    eta_minutes = remaining_batches * seconds_per_batch / 60
                    print(f"  [{stage_name}] epoch {global_epoch} batch {batch_idx + 1}/{len(train_loader)} "
                          f"rss_mb={current_rss_mb():.0f} sec_per_batch={seconds_per_batch:.3f} "
                          f"epoch_eta_min={eta_minutes:.1f}")
                    last_checkpoint_time = now
                    last_checkpoint_batch = batch_idx + 1

            release_device_memory(device)
            val_metrics = evaluate(model, val_loader, label_names, device)
            print(f"[{stage_name}] epoch {global_epoch}: train_loss={epoch_loss / max(n_batches, 1):.4f} "
                  f"val_macro_f1={val_metrics.macro_f1:.4f} rss_mb={current_rss_mb():.0f}")

            if val_metrics.macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_metrics.macro_f1
                best_state = {
                    "backbone": {k: v.clone() for k, v in model.backbone.state_dict().items()},
                    "pooling": {k: v.clone() for k, v in model.pooling.state_dict().items()},
                    "head": {k: v.clone() for k, v in model.head.state_dict().items()},
                }
                best_epoch = global_epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"[{stage_name}] early stopping at epoch {global_epoch} (no improvement for {patience} epochs).")
                    save_resume_checkpoint(
                        resume_path, stage_name=stage_name, epoch_in_stage=epoch_in_stage + 1, model=model,
                        optimizer=optimizer, best_val_macro_f1=best_val_macro_f1, best_state=best_state,
                        best_epoch=best_epoch, global_epoch=global_epoch,
                        epochs_without_improvement=epochs_without_improvement,
                    )
                    break

            # Saved after every completed epoch (not just on improvement) -
            # see save_resume_checkpoint's docstring. Cheap relative to an
            # epoch's own duration; written to a local temp path, removed
            # once training finishes successfully (see end of run_training).
            save_resume_checkpoint(
                resume_path, stage_name=stage_name, epoch_in_stage=epoch_in_stage + 1, model=model,
                optimizer=optimizer, best_val_macro_f1=best_val_macro_f1, best_state=best_state,
                best_epoch=best_epoch, global_epoch=global_epoch,
                epochs_without_improvement=epochs_without_improvement,
            )

    # Resuming into stage2 means stage1 already fully completed in a prior
    # (killed/crashed) run - re-running it would both waste time and
    # contradict the saved best_val_macro_f1/best_state, which already
    # reflects whichever stage1 epoch was actually best. Only the
    # architecture transition (freeze -> unfreeze) needs to be replayed,
    # never the training itself.
    resuming_into_stage2 = resume_data is not None and resume_data["stage_name"] == "stage2"

    # Stage 1: frozen backbone, train pooling + head only. No gradient
    # checkpointing here - the entire backbone is frozen, so no backward
    # pass through it happens regardless (see gradient_checkpointing_enable
    # below for why it's deliberately NOT enabled at this point).
    model.freeze_backbone()
    stage1_optimizer = torch.optim.AdamW(
        list(model.pooling.parameters()) + list(model.head.parameters()), lr=args.stage1_lr, weight_decay=1e-4
    )
    if resuming_into_stage2:
        print("Resume target is stage2 - skipping stage1 training (already complete in the resumed run).")
    else:
        stage1_start_epoch = 0
        stage1_initial_patience_count = 0
        if resume_data is not None and resume_data["stage_name"] == "stage1":
            stage1_optimizer.load_state_dict(resume_data["optimizer_state_dict"])
            stage1_start_epoch = resume_data["epoch_in_stage"]
            stage1_initial_patience_count = resume_data["epochs_without_improvement"]
        train_one_stage(
            "stage1", args.stage1_epochs, stage1_optimizer, args.early_stopping_patience,
            grad_accum_steps=args.grad_accum_steps, start_epoch_in_stage=stage1_start_epoch,
            initial_epochs_without_improvement=stage1_initial_patience_count,
        )

    # Stage 2: unfreeze the top N transformer layers at a much smaller LR;
    # pooling+head continue at a smaller (not stage-1) LR - see module docstring.
    model.unfreeze_top_layers(args.unfreeze_layers)

    if args.gradient_checkpointing:
        # Enabled starting here, not from the beginning of the script:
        # Hugging Face's `gradient_checkpointing_enable()` is a coarse,
        # whole-encoder switch - it would wrap every one of the 12
        # transformer layers' forward pass in torch.utils.checkpoint,
        # including the (still, even in stage 2) frozen bottom layers,
        # which need no backward pass at all and gain nothing from
        # checkpointing. That's a real, accepted, honestly-documented
        # inefficiency of using HF's API at this granularity (a from-
        # scratch custom per-layer checkpointing scheme could avoid it, but
        # isn't implemented here - not needed to prove the memory fix
        # works). The real benefit this provides: the `--unfreeze-layers`
        # top layers' *own* internal attention/feed-forward activations
        # (needed for their own weight gradients) are recomputed during
        # backward instead of held in memory for the whole forward pass -
        # trading recompute for memory, which is exactly the lever this
        # redesign needs on an 8GB machine. Never enabled in stage 1 (see
        # above) since there's nothing there for it to help with.
        model.backbone.gradient_checkpointing_enable()
        print(f"Gradient checkpointing enabled for stage 2 ({args.unfreeze_layers} unfrozen layer(s)).")

    stage2_optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.stage2_backbone_lr},
            {"params": list(model.pooling.parameters()) + list(model.head.parameters()), "lr": args.stage2_head_lr},
        ],
        weight_decay=1e-4,
    )
    stage2_start_epoch = 0
    stage2_initial_patience_count = 0
    if resuming_into_stage2:
        # `resuming_into_stage2` already encodes `resume_data is not None`
        # (see its definition above), but that's a fact about a *different*
        # variable mypy can't carry across into this block - this assert
        # is a real, always-true-here check that makes the dependency
        # explicit rather than working around it with a cast.
        assert resume_data is not None
        stage2_optimizer.load_state_dict(resume_data["optimizer_state_dict"])
        stage2_start_epoch = resume_data["epoch_in_stage"]
        stage2_initial_patience_count = resume_data["epochs_without_improvement"]
    train_one_stage(
        "stage2", args.stage2_epochs, stage2_optimizer, args.early_stopping_patience,
        grad_accum_steps=args.grad_accum_steps, start_epoch_in_stage=stage2_start_epoch,
        initial_epochs_without_improvement=stage2_initial_patience_count,
    )

    training_duration_seconds = time.monotonic() - training_started

    if best_state is None:
        print("error: training produced no valid checkpoint.", file=sys.stderr)
        sys.exit(1)
    model.backbone.load_state_dict(best_state["backbone"])
    model.pooling.load_state_dict(best_state["pooling"])
    model.head.load_state_dict(best_state["head"])
    model.to(device)

    test_metrics = evaluate(model, test_loader, label_names, device)
    print("\n=== Final TEST set evaluation (held out, never used for training/model selection) ===")
    print(f"accuracy={test_metrics.accuracy:.4f} macro_f1={test_metrics.macro_f1:.4f} "
          f"weighted_f1={test_metrics.weighted_f1:.4f} balanced_accuracy={test_metrics.balanced_accuracy:.4f}")

    # Per-dataset-source breakdown on the test split (docs/emotion.md,
    # "Cross-dataset evaluation") - computed here (not a separate script
    # invocation) since the trained model and test_loader/labels are
    # already in memory.
    per_dataset_metrics: dict[str, dict] = {}
    for dataset_source in sorted({s.dataset_source for s in test_samples}):
        subset_samples = [s for s in test_samples if s.dataset_source == dataset_source]
        subset_ds = RawWaveformDataset(
            subset_samples, cache_dir=cache_dir, label_names=label_names, augment=False,
            augmentation_config=augmentation_config, seed=args.seed, max_audio_seconds=args.max_audio_seconds,
        )
        subset_loader = DataLoader(subset_ds, batch_size=args.batch_size, collate_fn=collate_fn)
        subset_metrics = evaluate(model, subset_loader, label_names, device)
        per_dataset_metrics[dataset_source] = subset_metrics.to_dict()
        print(f"  [{dataset_source} test subset, n={subset_metrics.n_samples}] "
              f"accuracy={subset_metrics.accuracy:.4f} macro_f1={subset_metrics.macro_f1:.4f}")

    model.eval()
    checkpoint = EmotionCheckpointV2.from_model(
        model,
        provenance={
            "seed": args.seed,
            "stage1_epochs_requested": args.stage1_epochs,
            "stage2_epochs_requested": args.stage2_epochs,
            "best_epoch": best_epoch,
            "stage1_lr": args.stage1_lr,
            "stage2_backbone_lr": args.stage2_backbone_lr,
            "stage2_head_lr": args.stage2_head_lr,
            "unfreeze_layers": args.unfreeze_layers,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            "focal_gamma": args.focal_gamma,
            "label_smoothing": args.label_smoothing,
            "augmentation": not args.no_augmentation,
            "restrict_train_to_dataset": args.restrict_train_to_dataset,
            "training_duration_seconds": training_duration_seconds,
            "hardware": platform.platform(),
            "device": device,
            "library_versions": {"torch": torch.__version__, "transformers": transformers_version},
        },
    )
    storage_key = f"{settings.EMOTION_MODEL_ARTIFACT_ROOT}/{args.version_tag}/classifier.pt"
    storage = build_storage_backend(settings)
    checkpoint_bytes = checkpoint.to_bytes()
    await storage.upload(storage_key, checkpoint_bytes, content_type="application/octet-stream")
    print(f"\nSaved checkpoint to storage key: {storage_key} ({len(checkpoint_bytes) / 1e6:.1f} MB)")

    mlflow_run_id = None
    try:
        import mlflow

        with mlflow.start_run(run_name=args.version_tag) as run:
            mlflow.log_params(checkpoint.provenance)
            mlflow.log_metric("val_best_macro_f1", best_val_macro_f1)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.to_dict().items() if isinstance(v, (int, float))})
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
            training_config={"architecture": "v2-wav2vec2-attention", **checkpoint.provenance},
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
            },
            mlflow_run_id=mlflow_run_id,
            is_active=False,
        )
        if args.activate:
            await repo.activate(version.id)
        await session.commit()
        print(f"\nRegistered model_versions row: {version.id} (version_tag={args.version_tag})")
        print("Activated as the default emotion_classifier model." if args.activate else
              "Not activated - review the evaluation above before activating.")

    # Resume state is only useful until a run finishes successfully -
    # removed here so a later, unrelated run doesn't accidentally resume
    # from a stale, already-completed training's state.
    if resume_path.exists():
        resume_path.unlink()


def _probe_hidden_dim(base_model: str) -> int:
    """Reads the backbone's real hidden size from its published config
    (no weight download) before constructing EmotionClassifierV2Config -
    Wav2VecAttentionEmotionNet.__init__ re-derives and validates the same
    value itself when it builds the actual model."""
    from transformers import Wav2Vec2Config

    return Wav2Vec2Config.from_pretrained(base_model).hidden_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--version-tag", required=True)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cpu", choices=["auto", "cpu", "mps", "cuda"],
        help="Defaults to cpu, not auto/mps: a real full-scale training attempt on this "
        "project's actual hardware (8GB-unified-memory Apple M2) demonstrated severe, "
        "sustained MPS-related memory growth and swap thrashing (docs/emotion.md's "
        "'Training safety' section) - CPU avoids Metal's separate buffer-allocation "
        "behavior entirely. Pass --device mps or --device auto to opt back in.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Micro-batch size - kept deliberately tiny for real, demonstrated memory "
        "reasons (see --device's help). Use --grad-accum-steps to reach a reasonable "
        "effective batch size without increasing this.",
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=8,
        help="Gradients from this many micro-batches are summed (loss scaled by "
        "1/grad_accum_steps first) before each optimizer step - effective batch size = "
        "batch_size * grad_accum_steps, without ever holding more than one micro-batch's "
        "activations in memory at once.",
    )
    parser.add_argument(
        "--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=True,
        help="Enabled by default for stage 2 only (trades recompute for memory on the "
        "unfrozen top layers) - see the comment at its call site for the honest trade-off "
        "of using Hugging Face's whole-encoder API for this. Use --no-gradient-checkpointing to disable.",
    )
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-backbone-lr", type=float, default=2e-5)
    parser.add_argument("--stage2-head-lr", type=float, default=1e-4)
    parser.add_argument(
        "--unfreeze-layers", type=int, default=1,
        help="Reduced from an earlier default of 2 to 1 - fewer trainable backbone "
        "layers means less optimizer state and backward-pass memory, part of this same "
        "memory-conservative redesign. A hyperparameter, not an architecture change - "
        "unfreeze_top_layers() itself is unchanged and still supports any value.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--max-audio-seconds", type=float, default=MAX_AUDIO_SECONDS_DEFAULT)
    parser.add_argument(
        "--waveform-cache-dir", default="/tmp/voxmind_emotion_v2_waveform_cache",
        help="Where preprocessed (mono, 16kHz) waveforms are cached as .npy files - see "
        "preprocess_and_cache_waveforms's docstring for why this is disk, not memory.",
    )
    parser.add_argument(
        "--restrict-train-to-dataset", default=None, choices=["ravdess", "crema_d"],
        help="Cross-corpus diagnostic only: train on one dataset's train split, evaluate on the unrestricted test split.",
    )
    parser.add_argument(
        "--resume-checkpoint-path", default="/tmp/voxmind_emotion_v2_resume.pt",
        help="Where per-epoch resume state is saved (see save_resume_checkpoint) - always written to, "
        "regardless of --resume, so a later --resume can pick up from it.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load --resume-checkpoint-path (if it exists) and continue from exactly the last completed "
        "epoch, instead of starting fresh from the downloaded pretrained backbone. Added after a real "
        "full-scale run stalled for hours and had to be killed with no way to recover already-completed "
        "epochs - see docs/emotion.md's 'Training safety' section.",
    )
    args = parser.parse_args()
    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
