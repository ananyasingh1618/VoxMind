"""The trainable classification head on top of frozen emotion2vec+
embeddings, plus checkpoint save/load - mirrors v1's `model.py` and v2's
`v2/model.py` philosophy exactly (one architecture definition, shared by
the offline training script and the inference-time provider; a checkpoint
bundles weights + config + label mapping as one artifact, never weights
alone).

Architecture (per docs/emotion.md's design, Stage A): a deliberately
small, defensible head - emotion2vec+'s own 768-dim pretrained
representation already does the heavy representational work (this is the
same reasoning v1's docstring gives for its own small MLP on top of a
frozen embedding).

    768-dim embedding -> LayerNorm -> Linear(768, 256) -> GELU -> Dropout -> Linear(256, 6)
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field

import torch
from torch import nn


@dataclass
class Emotion2VecHeadConfig:
    embedding_dim: int
    label_names: list[str]
    hidden_dim: int = 256
    dropout: float = 0.3

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


class Emotion2VecHead(nn.Module):
    def __init__(self, config: Emotion2VecHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """`embedding` is the frozen emotion2vec+ utterance-level vector,
        shape (batch, embedding_dim). Returns raw logits, shape
        (batch, num_classes) - callers apply softmax."""
        return self.net(embedding)


@dataclass
class Emotion2VecCheckpoint:
    """Bundles the trained head's weights with its config and label
    mapping - never the encoder's own weights (those are the fixed,
    publicly-published `emotion2vec_plus_base` checkpoint, reloaded via
    FunASR at inference time, not duplicated here - unlike v2's
    self-contained approach, this head is tiny and the pretrained encoder
    is large and already independently versioned/hosted, so re-bundling it
    would be pure waste, not added robustness)."""

    config: Emotion2VecHeadConfig
    state_dict: dict = field(repr=False)
    provenance: dict = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict, "provenance": self.provenance}, buffer)
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "Emotion2VecCheckpoint":
        buffer = io.BytesIO(data)
        payload = torch.load(buffer, map_location="cpu", weights_only=False)
        return cls(
            config=Emotion2VecHeadConfig(**payload["config"]),
            state_dict=payload["state_dict"],
            provenance=payload.get("provenance", {}),
        )

    def build_model(self) -> Emotion2VecHead:
        model = Emotion2VecHead(self.config)
        model.load_state_dict(self.state_dict)
        model.eval()
        return model
