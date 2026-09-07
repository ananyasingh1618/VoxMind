"""Extends test_active_classifier_v2_routing.py's coverage to the
emotion2vec+ architecture tag - proves `load_active_classifier` correctly
resolves an `Emotion2VecClassifier` + paired `Emotion2VecEncoder` when the
active `ModelVersion` is tagged `architecture=="emotion2vec-plus-base"`,
exactly the same "one active row per component" mechanism v1/v2 already
use (model_version_repository.py, unmodified)."""
from __future__ import annotations

import pytest

from voxmind.core.config import get_settings
from voxmind.repositories.model_version_repository import ModelVersionRepository
from voxmind.services.emotion.active_classifier import load_active_classifier
from voxmind.services.emotion.emotion2vec.classifier_provider import Emotion2VecClassifier
from voxmind.services.emotion.emotion2vec.encoder import EMBEDDING_DIM, Emotion2VecEncoder
from voxmind.services.emotion.emotion2vec.model import Emotion2VecCheckpoint, Emotion2VecHead, Emotion2VecHeadConfig
from voxmind.services.storage.factory import build_storage_backend

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


async def _register_emotion2vec(session, storage) -> str:
    config = Emotion2VecHeadConfig(embedding_dim=EMBEDDING_DIM, label_names=LABELS, hidden_dim=32)
    model = Emotion2VecHead(config)
    checkpoint = Emotion2VecCheckpoint(config=config, state_dict=model.state_dict(), provenance={"note": "test-only, untrained"})
    storage_key = "models/emotion/test-emotion2vec-routing/classifier.pt"
    await storage.upload(storage_key, checkpoint.to_bytes())

    repo = ModelVersionRepository(session)
    version = await repo.create(
        component="emotion_classifier", version_tag="test-emotion2vec-routing", trained=True,
        artifact_storage_key=storage_key, training_config={"architecture": "emotion2vec-plus-base"},
        is_active=True,
    )
    return str(version.id)


@pytest.mark.asyncio
async def test_emotion2vec_model_resolves_to_emotion2vec_classifier_with_a_paired_encoder(db_session):
    settings = get_settings()
    storage = build_storage_backend(settings)
    version_id = await _register_emotion2vec(db_session, storage)
    await db_session.commit()

    result = await load_active_classifier(db_session, storage, settings=settings)

    assert result is not None
    classifier, model_version, provider_override = result
    assert isinstance(classifier, Emotion2VecClassifier)
    assert str(model_version.id) == version_id
    assert isinstance(provider_override, Emotion2VecEncoder)
    await db_session.commit()
