"""Pipeline stages implementing the approved, unmodified `PipelineStage`
contract (workers/task_runner.py) - same pattern as Phase 2's
services/speech/stages.py. Each stage takes/returns plain, JSON-serializable
Pydantic models; no torch.Tensor, numpy.ndarray, or Hugging Face model
object crosses a stage boundary.

Training is intentionally NOT a PipelineStage - it's an offline ML workflow
(ml/training/train_emotion_model.py), not something dispatched through the
online TaskRunner, per the approved architecture.
"""
from __future__ import annotations

from pydantic import BaseModel

from voxmind.services.emotion.audio_slicing import slice_wav_segment
from voxmind.services.emotion.interfaces import (
    AcousticFeatureExtractor,
    AcousticFeatures,
    EmotionClassifier,
    EmotionPrediction,
    SpeechEmbedding,
    Wav2Vec2EmbeddingProvider,
)
from voxmind.services.storage.interfaces import StorageBackend


class AcousticFeatureInput(BaseModel):
    processed_storage_key: str
    start_ms: int
    end_ms: int


class AcousticFeatureOutput(BaseModel):
    features: AcousticFeatures


class AcousticFeatureExtractionStage:
    name = "acoustic_feature_extraction"

    def __init__(self, storage: StorageBackend, extractor: AcousticFeatureExtractor) -> None:
        self._storage = storage
        self._extractor = extractor

    async def run(self, input: AcousticFeatureInput) -> AcousticFeatureOutput:
        wav_bytes = await self._storage.download(input.processed_storage_key)
        segment_bytes, sample_rate = slice_wav_segment(
            wav_bytes, start_ms=input.start_ms, end_ms=input.end_ms
        )
        features = await self._extractor.extract(segment_bytes, sample_rate=sample_rate)
        return AcousticFeatureOutput(features=features)


class EmbeddingInput(BaseModel):
    processed_storage_key: str
    start_ms: int
    end_ms: int


class EmbeddingOutput(BaseModel):
    embedding: SpeechEmbedding


class Wav2Vec2EmbeddingStage:
    name = "wav2vec2_embedding"

    def __init__(self, storage: StorageBackend, provider: Wav2Vec2EmbeddingProvider) -> None:
        self._storage = storage
        self._provider = provider

    async def run(self, input: EmbeddingInput) -> EmbeddingOutput:
        wav_bytes = await self._storage.download(input.processed_storage_key)
        segment_bytes, sample_rate = slice_wav_segment(
            wav_bytes, start_ms=input.start_ms, end_ms=input.end_ms
        )
        embedding = await self._provider.embed(segment_bytes, sample_rate=sample_rate)
        return EmbeddingOutput(embedding=embedding)


class EmotionInferenceInput(BaseModel):
    features: AcousticFeatures
    embedding: SpeechEmbedding


class EmotionInferenceOutput(BaseModel):
    prediction: EmotionPrediction


class EmotionInferenceStage:
    name = "emotion_inference"

    def __init__(self, classifier: EmotionClassifier) -> None:
        # Typed to the existing, unmodified `EmotionClassifier` Protocol
        # (interfaces.py) rather than v1's concrete `TorchEmotionClassifier`
        # - this stage only ever calls `.predict(features, embedding)`, so
        # it was already structurally duck-typed; this makes that explicit
        # so Emotion Model v2's `TorchEmotionClassifierV2`
        # (services/emotion/v2/inference.py) - which also implements the
        # same Protocol - can be passed here too, without this stage's
        # behavior changing at all for v1.
        self._classifier = classifier

    async def run(self, input: EmotionInferenceInput) -> EmotionInferenceOutput:
        prediction = await self._classifier.predict(input.features, input.embedding)
        return EmotionInferenceOutput(prediction=prediction)
