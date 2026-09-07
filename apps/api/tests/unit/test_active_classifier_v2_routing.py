"""Tests `active_classifier.py`'s v1/v2 routing logic directly (no
existing test covered this file at all before Emotion Model v2) - the
`training_config["architecture"]` convention that lets one `ModelVersion`
row resolve to v1's frozen-embedding classifier and another resolve to
v2's fine-tuned-backbone classifier+provider pair, entirely through
`get_active_for_component`'s existing "one active row per component"
semantics (model_version_repository.py, unmodified)."""
from __future__ import annotations

import pytest

from voxmind.core.config import get_settings
from voxmind.repositories.model_version_repository import ModelVersionRepository
from voxmind.services.emotion.active_classifier import load_active_classifier
from voxmind.services.emotion.classifier_provider import TorchEmotionClassifier
from voxmind.services.emotion.model import EmotionCheckpoint, EmotionClassifierConfig, EmotionClassifierNet, FeatureNormalizer
from voxmind.services.emotion.v2.inference import AttentionWav2Vec2Provider, TorchEmotionClassifierV2
from voxmind.services.emotion.v2.model import EmotionCheckpointV2, EmotionClassifierV2Config, Wav2VecAttentionEmotionNet
from voxmind.services.storage.factory import build_storage_backend

V1_LABELS = ["neutral", "happy", "sad", "angry"]
V2_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
BASE_MODEL = "facebook/wav2vec2-base"


async def _register_v1(session, storage) -> str:
    config = EmotionClassifierConfig(acoustic_dim=134, embedding_dim=768, label_names=V1_LABELS)
    model = EmotionClassifierNet(config)
    normalizer = FeatureNormalizer(mean=[0.0] * config.input_dim, std=[1.0] * config.input_dim)
    checkpoint = EmotionCheckpoint(config=config, normalizer=normalizer, state_dict=model.state_dict())
    storage_key = "models/emotion/test-v1-routing/classifier.pt"
    await storage.upload(storage_key, checkpoint.to_bytes())

    repo = ModelVersionRepository(session)
    version = await repo.create(
        component="emotion_classifier", version_tag="test-v1-routing", trained=True,
        artifact_storage_key=storage_key, training_config={"some": "v1 config, no architecture key"},
        is_active=True,
    )
    return str(version.id)


async def _register_v2(session, storage) -> str:
    from transformers import Wav2Vec2Config

    hidden_size = Wav2Vec2Config.from_pretrained(BASE_MODEL).hidden_size
    config = EmotionClassifierV2Config(base_model=BASE_MODEL, backbone_hidden_dim=hidden_size, label_names=V2_LABELS)
    model = Wav2VecAttentionEmotionNet(config)
    checkpoint = EmotionCheckpointV2.from_model(model, provenance={"note": "test-only, untrained"})
    storage_key = "models/emotion/test-v2-routing/classifier.pt"
    await storage.upload(storage_key, checkpoint.to_bytes())

    repo = ModelVersionRepository(session)
    version = await repo.create(
        component="emotion_classifier", version_tag="test-v2-routing", trained=True,
        artifact_storage_key=storage_key, training_config={"architecture": "v2-wav2vec2-attention"},
        is_active=True,
    )
    return str(version.id)


@pytest.mark.asyncio
async def test_v1_model_resolves_to_v1_classifier_with_no_provider_override(db_session):
    settings = get_settings()
    storage = build_storage_backend(settings)
    version_id = await _register_v1(db_session, storage)
    await db_session.commit()

    result = await load_active_classifier(db_session, storage, settings=settings)

    assert result is not None
    classifier, model_version, provider_override = result
    assert isinstance(classifier, TorchEmotionClassifier)
    assert str(model_version.id) == version_id
    assert provider_override is None  # v1 callers keep using their own fixed, shared provider
    await db_session.commit()


@pytest.mark.asyncio
async def test_v2_model_resolves_to_v2_classifier_with_a_paired_provider(db_session):
    settings = get_settings()
    storage = build_storage_backend(settings)
    version_id = await _register_v2(db_session, storage)
    await db_session.commit()

    result = await load_active_classifier(db_session, storage, settings=settings)

    assert result is not None
    classifier, model_version, provider_override = result
    assert isinstance(classifier, TorchEmotionClassifierV2)
    assert str(model_version.id) == version_id
    assert isinstance(provider_override, AttentionWav2Vec2Provider)  # v2 needs its own fine-tuned backbone
    await db_session.commit()


@pytest.mark.asyncio
async def test_no_active_model_returns_none(db_session):
    settings = get_settings()
    storage = build_storage_backend(settings)

    result = await load_active_classifier(db_session, storage, settings=settings)

    assert result is None
    await db_session.commit()
