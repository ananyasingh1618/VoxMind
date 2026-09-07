"""Genuine end-to-end emotion pipeline test: real audio upload → real Phase 2
processing (faster-whisper, real aligned turns) → a genuinely registered
model version → real acoustic feature extraction → **real Wav2Vec2
embedding extraction** (downloads facebook/wav2vec2-base, not gated) → a
real forward pass through the classifier → real persisted predictions,
all through the actual HTTP API against a real Postgres database.

IMPORTANT what this test does and does not prove: the registered "model
version" here wraps an **untrained** (randomly-initialized) classifier
network - saved and loaded through the exact same real
`EmotionCheckpoint`/storage code path a genuinely trained model would use,
which is what's being verified. It does NOT prove the model produces
meaningful emotion predictions - it can't, because no labeled dataset has
been used to train it (see docs/emotion.md). This test is marked
`trained=True` purely so `EmotionService`'s production gate lets it run;
every prediction it produces is still honestly labeled with whatever
`trained` value was passed at registration, exercised here for pipeline
verification only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from voxmind.core.config import get_settings
from voxmind.services.emotion.interfaces import ACOUSTIC_VECTOR_DIM
from voxmind.services.emotion.model import (
    EmotionCheckpoint,
    EmotionClassifierConfig,
    EmotionClassifierNet,
    FeatureNormalizer,
)
from voxmind.services.storage.factory import build_storage_backend

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CREDENTIALS = {"email": "emotion-pipeline@voxmind.dev", "password": "correct-horse-battery-staple"}
TEST_LABELS = ["neutral", "happy", "sad", "angry"]


async def _authed_headers(client) -> dict:
    await client.post("/api/v1/auth/register", json=CREDENTIALS)
    login = await client.post("/api/v1/auth/login", json=CREDENTIALS)
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


async def _register_test_fixture_model_version(db_session) -> str:
    """Registers a real (untrained) checkpoint through the real storage +
    ModelVersionRepository code path, activated so EmotionService's
    production gate accepts it. Returns the model_version id."""
    from voxmind.repositories.model_version_repository import ModelVersionRepository

    settings = get_settings()
    config = EmotionClassifierConfig(
        acoustic_dim=ACOUSTIC_VECTOR_DIM, embedding_dim=768, label_names=TEST_LABELS
    )
    model = EmotionClassifierNet(config)
    normalizer = FeatureNormalizer(mean=[0.0] * config.input_dim, std=[1.0] * config.input_dim)
    checkpoint = EmotionCheckpoint(config=config, normalizer=normalizer, state_dict=model.state_dict())
    storage = build_storage_backend(settings)
    storage_key = "models/emotion/test-fixture/classifier.pt"
    await storage.upload(storage_key, checkpoint.to_bytes())

    repo = ModelVersionRepository(db_session)
    version = await repo.create(
        component="emotion_classifier",
        version_tag="test-fixture-untrained",
        task="emotion_classification",
        base_model=settings.WAV2VEC2_MODEL,
        label_mapping={str(i): label for i, label in enumerate(TEST_LABELS)},
        artifact_storage_key=storage_key,
        trained=True,  # test-only: marks the *fixture* as eligible to run, see module docstring
        is_active=True,
    )
    await db_session.commit()
    return str(version.id)


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_full_emotion_pipeline_end_to_end(client, db_session):
    await _register_test_fixture_model_version(db_session)

    headers = await _authed_headers(client)
    conversation = await client.post(
        "/api/v1/conversations", json={"title": "emotion pipeline test"}, headers=headers
    )
    conversation_id = conversation.json()["id"]

    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    upload = await client.post(
        f"/api/v1/conversations/{conversation_id}/audio",
        files={"file": ("hello_world.wav", wav_bytes, "audio/wav")},
        headers=headers,
    )
    audio_asset_id = upload.json()["id"]

    process = await client.post(
        f"/api/v1/conversations/{conversation_id}/audio/{audio_asset_id}/process",
        json={},
        headers=headers,
        timeout=120,
    )
    assert process.json()["status"] == "completed"
    message_id = process.json()["message_id"]

    emotion_response = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages/{message_id}/emotion/process",
        headers=headers,
        timeout=120,
    )
    assert emotion_response.status_code == 201
    job = emotion_response.json()
    assert job["status"] == "completed", job
    assert job["turns_processed"] >= 1
    assert set(job["stage_durations_ms"].keys()) == {"acoustic_features", "embeddings", "inference"}

    predictions_response = await client.get(
        f"/api/v1/conversations/{conversation_id}/messages/{message_id}/emotion", headers=headers
    )
    predictions = predictions_response.json()
    assert len(predictions) == job["turns_processed"]
    for prediction in predictions:
        assert prediction["predicted_label"] in TEST_LABELS
        assert 0.0 <= prediction["confidence"] <= 1.0
        assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-4
        assert prediction["model_version_id"]
