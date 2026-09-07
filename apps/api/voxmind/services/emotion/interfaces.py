"""Emotion service boundary: acoustic features, Wav2Vec2 embeddings, and the
trainable classifier that combines them.

`trained: bool` on `EmotionPrediction` is load-bearing, not decorative:
until a real training run (ml/training/train_emotion_model.py) has produced
a checkpoint and been evaluated (ml/evaluation/evaluate_emotion_model.py),
no `EmotionPrediction` is served through the product-facing API at all -
see `services/emotion_service.py` for the gate. The untrained-baseline path
exists only for testing that the architecture/pipeline plumbing works
(`tests/*/test_emotion_*.py`), and is never reachable from
`api/v1/endpoints/emotion.py`.

Everything here is plain, JSON-serializable data - no `torch.Tensor`,
`numpy.ndarray`, or Hugging Face model object crosses this boundary.
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class AcousticStat(BaseModel):
    """Aggregated (not frame-level) statistics for one acoustic descriptor
    across all analyzed frames - the classifier consumes these, not raw
    frame sequences."""

    mean: float
    std: float
    min: float
    max: float
    p25: float
    p50: float
    p75: float


class AcousticFeatures(BaseModel):
    """Real, signal-derived acoustic descriptors - evidence the classifier
    may use, never an emotion label or a claim about emotion itself. See
    docs/emotion.md for exactly how each is computed and why some
    plausible-sounding features (e.g. a music-style tempo/"speech rate"
    estimate) were deliberately left out rather than misapplied to speech.
    """

    duration_seconds: float
    voiced_fraction: float = Field(description="Fraction of frames with a detected pitch.")
    pitch_hz: AcousticStat
    rms_energy: AcousticStat
    zero_crossing_rate: AcousticStat
    spectral_centroid: AcousticStat
    spectral_bandwidth: AcousticStat
    spectral_rolloff: AcousticStat
    mfcc: list[AcousticStat] = Field(description="One entry per MFCC coefficient, in order.")

    def to_vector(self) -> list[float]:
        """Deterministic flattening into a fixed-length feature vector for
        the classifier - order must never change without bumping every
        model version's `config.acoustic_feature_version`."""
        stats = [
            self.pitch_hz,
            self.rms_energy,
            self.zero_crossing_rate,
            self.spectral_centroid,
            self.spectral_bandwidth,
            self.spectral_rolloff,
        ]
        vector = [self.voiced_fraction]
        for stat in stats:
            vector.extend([stat.mean, stat.std, stat.min, stat.max, stat.p25, stat.p50, stat.p75])
        for mfcc_stat in self.mfcc:
            vector.extend(
                [mfcc_stat.mean, mfcc_stat.std, mfcc_stat.min, mfcc_stat.max, mfcc_stat.p25, mfcc_stat.p50, mfcc_stat.p75]
            )
        return vector


ACOUSTIC_FEATURE_VERSION = "acoustic-v1"
# to_vector() length for the current extractor config (13 MFCCs): 1 + 6*7 + 13*7 = 134
ACOUSTIC_VECTOR_DIM = 1 + 6 * 7 + 13 * 7


class SpeechEmbedding(BaseModel):
    """A fixed-size, pooled Wav2Vec2 embedding - never the raw frame-level
    hidden states (see docs/emotion.md for the pooling strategy)."""

    vector: list[float]
    dim: int
    model_id: str
    pooling: str


class EmotionPrediction(BaseModel):
    label: str
    probabilities: dict[str, float]
    model_version_id: str
    trained: bool


class AcousticFeatureExtractor(Protocol):
    async def extract(self, audio_bytes: bytes, *, sample_rate: int) -> AcousticFeatures: ...


class Wav2Vec2EmbeddingProvider(Protocol):
    async def embed(self, audio_bytes: bytes, *, sample_rate: int) -> SpeechEmbedding: ...


class EmotionClassifier(Protocol):
    async def predict(
        self, features: AcousticFeatures, embedding: SpeechEmbedding
    ) -> EmotionPrediction: ...
