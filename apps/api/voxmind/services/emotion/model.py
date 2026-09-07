"""The trainable emotion classifier: architecture, feature normalization,
and checkpoint save/load. Used by both the offline training script
(ml/training/train_emotion_model.py) and the inference-time provider
(classifier_provider.py) - one definition, never duplicated.

Architecture (documented per project requirement): the real acoustic
feature vector (see acoustic_features.py, `ACOUSTIC_VECTOR_DIM` dims) is
concatenated with the pooled Wav2Vec2 embedding (768 dims for
wav2vec2-base), normalized (z-score, using statistics fit on the training
split only), and passed through a small MLP: Linear -> ReLU -> Dropout ->
Linear -> logits. This is a deliberately simple architecture appropriate
for a small-to-medium labeled speech-emotion dataset (e.g. RAVDESS's ~1440
clips) - a deeper network would be more prone to overfitting on a dataset
that size, and the acoustic+embedding feature fusion already does the heavy
representational lifting via Wav2Vec2's pretrained encoder.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field

import torch
from torch import nn


@dataclass
class EmotionClassifierConfig:
    acoustic_dim: int
    embedding_dim: int
    label_names: list[str]
    hidden_dim: int = 128
    dropout: float = 0.3

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    @property
    def input_dim(self) -> int:
        return self.acoustic_dim + self.embedding_dim


@dataclass
class FeatureNormalizer:
    """Z-score normalization fit on the training split only - applying
    statistics derived from validation/test data would leak information
    across the split boundary."""

    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, vectors: list[list[float]]) -> "FeatureNormalizer":
        tensor = torch.tensor(vectors, dtype=torch.float32)
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0, unbiased=False)  # population std - the standard z-score convention
        std = torch.where(std < 1e-6, torch.ones_like(std), std)  # avoid divide-by-zero on constant dims
        return cls(mean=mean.tolist(), std=std.tolist())

    def transform(self, vector: list[float]) -> torch.Tensor:
        tensor = torch.tensor(vector, dtype=torch.float32)
        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)
        return (tensor - mean) / std


class EmotionClassifierNet(nn.Module):
    def __init__(self, config: EmotionClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """`features` is the already-concatenated-and-normalized
        [acoustic; embedding] vector, shape (batch, input_dim). Returns raw
        logits, shape (batch, num_classes) - callers apply softmax."""
        return self.net(features)


@dataclass
class EmotionCheckpoint:
    """Everything needed to reconstruct and run a specific trained model
    version: weights, the exact normalization it was trained with, and its
    label mapping. Saved/loaded as one artifact through the storage
    abstraction - never as raw weights alone, which would be unusable
    without knowing the label order or normalization statistics."""

    config: EmotionClassifierConfig
    normalizer: FeatureNormalizer
    state_dict: dict = field(repr=False)

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {
                "config": asdict(self.config),
                "normalizer": asdict(self.normalizer),
                "state_dict": self.state_dict,
            },
            buffer,
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "EmotionCheckpoint":
        buffer = io.BytesIO(data)
        payload = torch.load(buffer, map_location="cpu", weights_only=False)
        return cls(
            config=EmotionClassifierConfig(**payload["config"]),
            normalizer=FeatureNormalizer(**payload["normalizer"]),
            state_dict=payload["state_dict"],
        )

    def build_model(self) -> EmotionClassifierNet:
        model = EmotionClassifierNet(self.config)
        model.load_state_dict(self.state_dict)
        model.eval()
        return model
