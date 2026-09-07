"""Unit tests for the emotion pipeline stages' own wiring logic - test
doubles for storage/providers/classifier, exactly like Phase 2's
test_speech_stages.py. Never used in the production AudioService/
EmotionService code paths, which always construct stages with the real
providers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from voxmind.services.emotion.classifier_provider import build_untrained_baseline
from voxmind.services.emotion.interfaces import ACOUSTIC_VECTOR_DIM, SpeechEmbedding
from voxmind.services.emotion.stages import (
    AcousticFeatureExtractionStage,
    AcousticFeatureInput,
    EmbeddingInput,
    EmotionInferenceInput,
    EmotionInferenceStage,
    Wav2Vec2EmbeddingStage,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class InMemoryStorage:
    def __init__(self, initial: dict[str, bytes] | None = None) -> None:
        self._data = dict(initial or {})

    async def upload(self, key, data, *, content_type=None):
        self._data[key] = data

    async def download(self, key):
        return self._data[key]

    async def delete(self, key):
        self._data.pop(key, None)

    async def exists(self, key):
        return key in self._data


class StubEmbeddingProvider:
    async def embed(self, audio_bytes: bytes, *, sample_rate: int) -> SpeechEmbedding:
        return SpeechEmbedding(vector=[0.1] * 768, dim=768, model_id="stub-wav2vec2", pooling="mean")


@pytest.mark.asyncio
async def test_acoustic_feature_stage_downloads_and_slices_real_audio():
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    storage = InMemoryStorage({"processed.wav": wav_bytes})

    from voxmind.services.emotion.acoustic_features import LibrosaAcousticFeatureExtractor

    stage = AcousticFeatureExtractionStage(storage, LibrosaAcousticFeatureExtractor())
    output = await stage.run(
        AcousticFeatureInput(processed_storage_key="processed.wav", start_ms=0, end_ms=1000)
    )

    assert len(output.features.to_vector()) == ACOUSTIC_VECTOR_DIM


@pytest.mark.asyncio
async def test_embedding_stage_downloads_and_forwards_to_provider():
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    storage = InMemoryStorage({"processed.wav": wav_bytes})

    stage = Wav2Vec2EmbeddingStage(storage, StubEmbeddingProvider())
    output = await stage.run(
        EmbeddingInput(processed_storage_key="processed.wav", start_ms=0, end_ms=1000)
    )

    assert output.embedding.dim == 768
    assert output.embedding.model_id == "stub-wav2vec2"


@pytest.mark.asyncio
async def test_emotion_inference_stage_runs_the_classifier():
    classifier = build_untrained_baseline(
        acoustic_dim=ACOUSTIC_VECTOR_DIM, embedding_dim=768, label_names=["neutral", "happy"]
    )
    stage = EmotionInferenceStage(classifier)

    from tests.unit.test_classifier_provider import _dummy_embedding, _dummy_features

    output = await stage.run(
        EmotionInferenceInput(features=_dummy_features(), embedding=_dummy_embedding())
    )

    assert output.prediction.trained is False
    assert output.prediction.label in ["neutral", "happy"]


def test_stage_names_are_explicit_and_distinct():
    names = {
        AcousticFeatureExtractionStage.name,
        Wav2Vec2EmbeddingStage.name,
        EmotionInferenceStage.name,
    }
    assert names == {"acoustic_feature_extraction", "wav2vec2_embedding", "emotion_inference"}
