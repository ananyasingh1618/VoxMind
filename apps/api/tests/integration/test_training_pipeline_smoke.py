"""Validates that the REAL training pipeline (ml/training/train_emotion_model.py)
runs end-to-end without error: real audio preprocessing, real acoustic
feature extraction, real Wav2Vec2 embedding extraction, a real training
loop, real checkpoint save through the storage abstraction, and real
`model_versions` registration in Postgres.

This is explicitly a PIPELINE MECHANICS test, not a model-quality test: the
"dataset" here is the same few seconds of synthesized speech (Phase 2's
`hello_world.wav` fixture) relabeled arbitrarily across fake speakers, since
no real labeled emotion corpus (RAVDESS) has been supplied to this
environment (see docs/emotion.md and ml/datasets/README.md). The resulting
`model_versions` row is real (real weights, real training config, really
persisted) but is NEVER claimed to be a genuine emotion classifier - it is
deliberately given `version_tag="pipeline-smoke-test"` and is never
activated, so `EmotionService`'s production gate (`is_active=True`) never
serves it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _build_smoke_test_manifest(tmp_path: Path):
    from ml.datasets.schema import DatasetSplit, EmotionSample, save_manifest

    wav_path = FIXTURES_DIR / "hello_world.wav"
    labels = ["happy", "sad", "neutral"]
    samples = []
    # 3 fake speakers, one per split, 3 samples each (all pointing at the
    # same real audio file with arbitrary labels - PIPELINE test, not a
    # claim about what emotion that clip actually expresses).
    for speaker_idx, split in enumerate([DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST]):
        for i, label in enumerate(labels):
            samples.append(
                EmotionSample(
                    sample_id=f"smoke-{speaker_idx}-{i}",
                    audio_path=str(wav_path),
                    label=label,
                    dataset_source="pipeline-smoke-test-not-a-real-dataset",
                    speaker_id=f"fake-speaker-{speaker_idx}",
                    split=split,
                )
            )
    manifest_path = tmp_path / "smoke_manifest.jsonl"
    save_manifest(samples, manifest_path)
    return manifest_path


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_training_pipeline_runs_end_to_end_and_registers_a_model_version(tmp_path, db_session):
    from ml.training.train_emotion_model import run_training
    from voxmind.repositories.model_version_repository import ModelVersionRepository

    manifest_path = _build_smoke_test_manifest(tmp_path)

    args = argparse.Namespace(
        manifest=str(manifest_path),
        version_tag="pipeline-smoke-test",
        activate=False,
        seed=42,
        learning_rate=1e-3,
        batch_size=2,
        epochs=2,
        hidden_dim=8,
        dropout=0.1,
        early_stopping_patience=5,
        no_class_weights=False,
    )

    await run_training(args)

    versions = await ModelVersionRepository(db_session).list_for_component("emotion_classifier")
    smoke_versions = [v for v in versions if v.version_tag == "pipeline-smoke-test"]
    assert len(smoke_versions) == 1

    version = smoke_versions[0]
    assert version.trained is True
    assert version.is_active is False  # never activated - not a real model
    assert version.artifact_storage_key is not None
    assert version.label_mapping == {"0": "happy", "1": "neutral", "2": "sad"}
    assert version.base_model
    assert version.metrics is not None
    assert "test" in version.metrics
    assert 0.0 <= version.metrics["test"]["accuracy"] <= 1.0
    await db_session.commit()
