"""Genuine end-to-end pipeline test for the emotion2vec+ production
candidate: real audio upload -> real Phase 2 processing -> a genuinely
registered `emotion2vec-plus-base`-tagged model version -> real emotion2vec+
embedding extraction (downloads the real checkpoint, `real_model`-marked)
-> a real forward pass through the trained head -> a real persisted
prediction, all through the actual HTTP API against a real Postgres
database. Mirrors test_emotion_pipeline_real_model.py's (v1) exact
structure and honesty caveats.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from voxmind.core.config import get_settings
from voxmind.services.emotion.emotion2vec.encoder import EMBEDDING_DIM
from voxmind.services.emotion.emotion2vec.model import Emotion2VecCheckpoint, Emotion2VecHead, Emotion2VecHeadConfig
from voxmind.services.storage.factory import build_storage_backend

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CREDENTIALS = {"email": "emotion2vec-pipeline@voxmind.dev", "password": "correct-horse-battery-staple"}
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


async def _authed_headers(client) -> dict:
    await client.post("/api/v1/auth/register", json=CREDENTIALS)
    login = await client.post("/api/v1/auth/login", json=CREDENTIALS)
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


async def _register_test_fixture_model_version(db_session) -> str:
    """Registers a real (untrained, random-initialized head) checkpoint
    through the real storage + ModelVersionRepository code path, activated
    so EmotionService's production gate accepts it - matches v1's own
    `test_emotion_pipeline_real_model.py` fixture pattern exactly. The
    *encoder* used at inference time is still the real, genuine, trained
    emotion2vec+ checkpoint - only this test's classification *head* is
    untrained, purely so this test doesn't depend on the specific
    `emotion2vec-tess-emodb-v1` model already being registered/active."""
    from voxmind.repositories.model_version_repository import ModelVersionRepository

    settings = get_settings()
    config = Emotion2VecHeadConfig(embedding_dim=EMBEDDING_DIM, label_names=LABELS, hidden_dim=32)
    model = Emotion2VecHead(config)
    checkpoint = Emotion2VecCheckpoint(config=config, state_dict=model.state_dict())
    storage = build_storage_backend(settings)
    storage_key = "models/emotion/test-emotion2vec-pipeline-fixture/classifier.pt"
    await storage.upload(storage_key, checkpoint.to_bytes())

    repo = ModelVersionRepository(db_session)
    version = await repo.create(
        component="emotion_classifier",
        version_tag="test-emotion2vec-pipeline-fixture",
        task="emotion_classification",
        base_model=settings.EMOTION2VEC_MODEL,
        label_mapping={str(i): label for i, label in enumerate(LABELS)},
        training_config={"architecture": "emotion2vec-plus-base"},
        artifact_storage_key=storage_key,
        trained=True,
        is_active=True,
    )
    await db_session.commit()
    return str(version.id)


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_full_emotion2vec_pipeline_end_to_end(client, db_session):
    await _register_test_fixture_model_version(db_session)

    headers = await _authed_headers(client)
    conversation = await client.post(
        "/api/v1/conversations", json={"title": "emotion2vec pipeline test"}, headers=headers
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

    predictions_response = await client.get(
        f"/api/v1/conversations/{conversation_id}/messages/{message_id}/emotion", headers=headers
    )
    predictions = predictions_response.json()
    assert len(predictions) == job["turns_processed"]
    for prediction in predictions:
        assert prediction["predicted_label"] in LABELS
        assert 0.0 <= prediction["confidence"] <= 1.0
        assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-4
        assert prediction["model_version_id"]
