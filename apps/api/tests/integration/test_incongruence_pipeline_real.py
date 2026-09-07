"""Genuine end-to-end incongruence test: real audio upload -> real Phase 2
transcription/alignment -> a registered (untrained-network, see
test_emotion_pipeline_real_model.py's docstring for why that's legitimate
here) emotion model -> real per-turn emotion predictions -> real per-turn
sentiment analysis on the turn's transcript text -> a genuine incongruence
signal combining both, all through the real HTTP API against a real
Postgres database.
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
CREDENTIALS = {"email": "incongruence-test@voxmind.dev", "password": "correct-horse-battery-staple"}
TEST_LABELS = ["neutral", "happy", "sad", "angry"]


async def _authed_headers(client) -> dict:
    await client.post("/api/v1/auth/register", json=CREDENTIALS)
    login = await client.post("/api/v1/auth/login", json=CREDENTIALS)
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


async def _register_test_fixture_model_version(db_session) -> None:
    from voxmind.repositories.model_version_repository import ModelVersionRepository

    settings = get_settings()
    config = EmotionClassifierConfig(
        acoustic_dim=ACOUSTIC_VECTOR_DIM, embedding_dim=768, label_names=TEST_LABELS
    )
    model = EmotionClassifierNet(config)
    normalizer = FeatureNormalizer(mean=[0.0] * config.input_dim, std=[1.0] * config.input_dim)
    checkpoint = EmotionCheckpoint(config=config, normalizer=normalizer, state_dict=model.state_dict())
    storage = build_storage_backend(settings)
    storage_key = "models/emotion/incongruence-test-fixture/classifier.pt"
    await storage.upload(storage_key, checkpoint.to_bytes())

    repo = ModelVersionRepository(db_session)
    await repo.create(
        component="emotion_classifier",
        version_tag="incongruence-test-fixture",
        task="emotion_classification",
        base_model=settings.WAV2VEC2_MODEL,
        label_mapping={str(i): label for i, label in enumerate(TEST_LABELS)},
        artifact_storage_key=storage_key,
        trained=True,
        is_active=True,
    )
    await db_session.commit()


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_incongruence_signal_is_genuinely_computed_from_real_upstream_signals(client, db_session):
    await _register_test_fixture_model_version(db_session)

    headers = await _authed_headers(client)
    conversation = await client.post(
        "/api/v1/conversations", json={"title": "incongruence test"}, headers=headers
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
    assert emotion_response.json()["status"] == "completed"

    incongruence_response = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages/{message_id}/incongruence/analyze",
        headers=headers,
        timeout=120,
    )
    assert incongruence_response.status_code == 201
    signals = incongruence_response.json()
    assert len(signals) >= 1
    for signal in signals:
        assert 0.0 <= signal["incongruence_score"] <= 1.0
        assert 0.0 <= signal["confidence"] <= 1.0
        assert signal["signal_category"] == "analytical_not_diagnostic"
        assert "not indicate deception" in signal["explanation"].lower()
        assert signal["semantic_signal"]["sentiment_label"] in ("positive", "neutral", "negative")
        assert signal["vocal_signal"]["predicted_label"] in TEST_LABELS

    fetched = await client.get(
        f"/api/v1/conversations/{conversation_id}/messages/{message_id}/incongruence", headers=headers
    )
    assert fetched.status_code == 200
    assert len(fetched.json()) == len(signals)


@pytest.mark.asyncio
async def test_incongruence_requires_authentication(client):
    response = await client.post(
        "/api/v1/conversations/00000000-0000-0000-0000-000000000000/messages/"
        "00000000-0000-0000-0000-000000000000/incongruence/analyze"
    )
    assert response.status_code == 401
