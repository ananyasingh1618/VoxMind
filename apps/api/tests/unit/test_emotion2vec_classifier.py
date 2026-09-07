"""Tests `Emotion2VecClassifier` (the head-only inference wrapper) without
needing the real emotion2vec+ encoder download - only the small trained
head is involved, so this stays a fast, real (not mocked) unit test,
mirroring v1/v2's own untrained-baseline-style pattern."""
from __future__ import annotations

import pytest

from voxmind.services.emotion.emotion2vec.classifier_provider import Emotion2VecClassifier
from voxmind.services.emotion.emotion2vec.model import Emotion2VecCheckpoint, Emotion2VecHead, Emotion2VecHeadConfig
from voxmind.services.emotion.interfaces import SpeechEmbedding

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


def _make_checkpoint() -> Emotion2VecCheckpoint:
    config = Emotion2VecHeadConfig(embedding_dim=768, label_names=LABELS, hidden_dim=32)
    model = Emotion2VecHead(config)
    return Emotion2VecCheckpoint(config=config, state_dict=model.state_dict())


@pytest.mark.asyncio
async def test_predict_ignores_features_and_uses_only_the_embedding():
    checkpoint = _make_checkpoint()
    classifier = Emotion2VecClassifier.from_checkpoint(checkpoint, model_version_id="test-e2v", trained=True)
    embedding = SpeechEmbedding(vector=[0.1] * 768, dim=768, model_id="emotion2vec/emotion2vec_plus_base", pooling="emotion2vec-utterance")

    prediction = await classifier.predict(features=object(), embedding=embedding)  # type: ignore[arg-type]

    assert prediction.label in LABELS
    assert prediction.model_version_id == "test-e2v"
    assert prediction.trained is True
    assert set(prediction.probabilities.keys()) == set(LABELS)
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-4


@pytest.mark.asyncio
async def test_predict_is_deterministic_for_the_same_embedding():
    checkpoint = _make_checkpoint()
    classifier = Emotion2VecClassifier.from_checkpoint(checkpoint, model_version_id="test-e2v", trained=True)
    embedding = SpeechEmbedding(vector=[0.3] * 768, dim=768, model_id="x", pooling="emotion2vec-utterance")

    p1 = await classifier.predict(features=object(), embedding=embedding)  # type: ignore[arg-type]
    p2 = await classifier.predict(features=object(), embedding=embedding)  # type: ignore[arg-type]

    assert p1.label == p2.label
    assert p1.probabilities == p2.probabilities


@pytest.mark.asyncio
async def test_untrained_flag_is_honestly_propagated():
    checkpoint = _make_checkpoint()
    classifier = Emotion2VecClassifier.from_checkpoint(checkpoint, model_version_id="test-e2v", trained=False)
    embedding = SpeechEmbedding(vector=[0.0] * 768, dim=768, model_id="x", pooling="emotion2vec-utterance")

    prediction = await classifier.predict(features=object(), embedding=embedding)  # type: ignore[arg-type]

    assert prediction.trained is False
