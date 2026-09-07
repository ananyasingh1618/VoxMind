"""Emotion Model v2's architecture (docs/emotion.md, section "Architecture"):

    raw waveform -> Wav2Vec2 (fine-tuned) -> hidden states
                 -> attention-weighted temporal pooling -> 768-dim vector
                 -> Linear(768, 256) -> ReLU -> Dropout -> Linear(256, 6)
                 -> emotion logits

This is a genuinely different architecture from v1's (frozen Wav2Vec2 mean-
pooled embedding concatenated with handcrafted acoustic features, feeding a
small MLP - see ../model.py) - not a variant of it. v1 is never imported
here and this module never imports v1's `EmotionClassifierNet`/
`EmotionCheckpoint`; the two are deliberately independent so nothing about
v1's frozen, already-registered `ravdess-v1` model can be affected by any
change here.

`Wav2VecAttentionEmotionNet` is the ONE nn.Module used for *training*
(bundles the backbone + pooling + head so a single optimizer can apply
per-group learning rates and freeze/unfreeze the backbone - see
ml/training/train_emotion_model_v2.py). At *inference* time, the same
weights are split across two small wrapper classes
(`v2/inference.py::AttentionWav2Vec2Provider` and `TorchEmotionClassifierV2`)
so the existing `Wav2Vec2EmbeddingProvider`/`EmotionClassifier` Protocols
(interfaces.py, unmodified) can serve v2 exactly like they already serve
v1 - see v2/inference.py's module docstring for why the split is at
exactly the embedding/classifier boundary.
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from typing import cast

import torch
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model


class AttentionPooling(nn.Module):
    """Learns which temporal frames of Wav2Vec2's `last_hidden_state`
    carry useful emotional information, instead of averaging every frame
    equally (v1's mean-pooling - see ../wav2vec_provider.py). A single
    linear layer scores each frame; scores are softmax-normalized over time
    (real frames only - padded frames are masked to -inf *before* the
    softmax, not after, so they receive exactly zero weight rather than a
    small nonzero one) and used to compute a weighted sum of the hidden
    states.

    Correctness note this project's v1 provider explicitly flagged as a
    future concern (wav2vec_provider.py's docstring): v1 only ever processes
    one unpadded clip at a time, so a plain mean is exactly correct there.
    v2 trains with real mini-batches of variable-length clips, which
    *does* require padding - so this pooling is mask-aware from the start,
    not a shortcut that would break under batching.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        hidden_states: (batch, time, hidden_dim)
        attention_mask: (batch, time), 1 for real frames / 0 for padding, or
            None when the caller guarantees no padding (e.g. batch size 1).
        Returns: (batch, hidden_dim)
        """
        scores = self.score(hidden_states).squeeze(-1)  # (batch, time)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (batch, time)
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden_dim)
        return pooled


@dataclass
class EmotionClassifierV2Config:
    base_model: str
    """The exact Wav2Vec2 checkpoint id used, e.g. "facebook/wav2vec2-base"
    - recorded so a checkpoint is never ambiguous about its backbone."""
    backbone_hidden_dim: int
    """Read from the loaded backbone's own config at construction time
    (see build_backbone_and_hidden_dim below) - never hardcoded to 768, so
    a differently-sized Wav2Vec2 checkpoint would still produce a
    dimensionally-correct model rather than a silent shape-mismatch crash."""
    label_names: list[str]
    head_hidden_dim: int = 256
    dropout: float = 0.3

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


def build_backbone(base_model: str) -> tuple[Wav2Vec2Model, int]:
    """Instantiates the Wav2Vec2 architecture from its published config
    (no weight download required - the actual weights this project uses,
    whether pretrained or later fine-tuned, always come from an explicit
    `load_state_dict` call, either from Hugging Face at the start of
    training or from our own self-contained checkpoint at inference/
    resume time - see EmotionCheckpointV2 below). Returns the model plus
    its real hidden size (not assumed to be 768)."""
    config = Wav2Vec2Config.from_pretrained(base_model)
    # facebook/wav2vec2-base's published config.json sets this legacy field
    # to True (confirmed in this environment - transformers itself warns
    # that passing it is deprecated). Left enabled, it wraps every encoder
    # layer's forward pass in torch.utils.checkpoint.checkpoint() whenever
    # the backbone module is in .train() mode - including layers this
    # project deliberately keeps frozen (the CNN feature extractor, and
    # whichever transformer layers stage 1/stage 2 haven't unfrozen yet -
    # see freeze_backbone()/unfreeze_top_layers() below), where no gradient
    # is needed at all. That produced a real, observed warning during the
    # mandatory smoke-training run ("None of the inputs have
    # requires_grad=True. Gradients will be None") and pure wasted compute
    # (checkpointing re-runs the forward pass during backward specifically
    # to save activation memory - pointless for a layer that needs no
    # backward pass through it in the first place). Explicitly disabled;
    # this project's batch sizes are already small for hardware-memory
    # reasons (docs/emotion.md, "Compute"), so checkpointing's memory/
    # compute trade-off was never being usefully spent here anyway.
    config.gradient_checkpointing = False
    model = Wav2Vec2Model(config)
    return model, config.hidden_size


class EmotionHead(nn.Module):
    def __init__(self, hidden_dim: int, head_hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class Wav2VecAttentionEmotionNet(nn.Module):
    """The full training-time module: backbone + attention pooling + head.
    `freeze_backbone()`/`unfreeze_top_layers()` implement the two-stage
    fine-tuning schedule from docs/emotion.md's "Training strategy"
    section; both always keep the convolutional feature-extractor frozen
    (`backbone.feature_extractor`), which is standard practice for Wav2Vec2
    fine-tuning (fine-tuning the raw-waveform CNN front-end is rarely
    beneficial and is far more memory/compute-expensive than fine-tuning
    only the transformer encoder layers on top of it) and, on this
    project's constrained hardware (an 8GB-unified-memory Apple M2, no
    CUDA - see docs/emotion.md's "Compute" section), a real practical
    requirement, not just a convention followed for its own sake."""

    def __init__(self, config: EmotionClassifierV2Config) -> None:
        super().__init__()
        self.config = config
        self.backbone, hidden_dim = build_backbone(config.base_model)
        if hidden_dim != config.backbone_hidden_dim:
            raise ValueError(
                f"backbone_hidden_dim in config ({config.backbone_hidden_dim}) does not match "
                f"the actual loaded backbone's hidden size ({hidden_dim})."
            )
        self.backbone.feature_extractor._requires_grad = False
        for p in self.backbone.feature_extractor.parameters():
            p.requires_grad = False
        self.pooling = AttentionPooling(hidden_dim)
        self.head = EmotionHead(hidden_dim, config.head_hidden_dim, config.num_classes, config.dropout)

    def freeze_backbone(self) -> None:
        """Stage 1 (docs/emotion.md): the entire Wav2Vec2 backbone is
        frozen; only pooling + head parameters receive gradients."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, n_layers: int) -> None:
        """Stage 2: unfreezes only the top `n_layers` transformer encoder
        layers (the ones closest to the task-specific head - standard
        progressive-unfreezing practice) plus the final `encoder.layer_norm`
        that follows them. The convolutional feature extractor and the
        positional-conv embedding stay frozen regardless of `n_layers`,
        for the memory/compute reasons in the class docstring."""
        encoder_layers = self.backbone.encoder.layers
        if not 0 < n_layers <= len(encoder_layers):
            raise ValueError(f"n_layers must be in (0, {len(encoder_layers)}], got {n_layers}.")
        for layer in encoder_layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.backbone.encoder.layer_norm.parameters():
            p.requires_grad = True

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Wav2Vec2Model's own attention_mask handles the CNN feature
        # extractor's downsampling internally (it derives the correct
        # frame-level mask from the sample-level one) - passed straight
        # through, not recomputed here.
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden_dim)

        pooling_mask = None
        if attention_mask is not None:
            pooling_mask = self.backbone._get_feature_vector_attention_mask(
                hidden_states.shape[1], cast(torch.LongTensor, attention_mask.long())
            )
        pooled = self.pooling(hidden_states, pooling_mask)
        return self.head(pooled)


@dataclass
class EmotionCheckpointV2:
    """Everything needed to reconstruct v2 exactly, bundled as one
    self-contained artifact - mirrors v1's `EmotionCheckpoint` philosophy
    (model.py) exactly: a checkpoint is unusable without its config and
    label mapping, so all three are always saved together. Unlike a
    "delta" checkpoint (saving only the fine-tuned layers and relying on a
    matching Hugging Face download at load time), this saves the FULL
    backbone state dict - larger on disk, but reproducible even if the
    upstream `facebook/wav2vec2-base` weights on the Hub were ever to
    change, and independent of Hub availability at inference time (only
    the small architecture *config*, not the weights, is fetched then -
    see build_backbone above)."""

    config: EmotionClassifierV2Config
    backbone_state_dict: dict = field(repr=False)
    pooling_state_dict: dict = field(repr=False)
    head_state_dict: dict = field(repr=False)
    provenance: dict = field(default_factory=dict)
    """Free-form, JSON-safe metadata recorded at training time: dataset
    counts/split/seed, hyperparameters, library versions, backbone
    revision - see docs/emotion.md and ml/training/train_emotion_model_v2.py."""

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {
                "config": asdict(self.config),
                "backbone_state_dict": self.backbone_state_dict,
                "pooling_state_dict": self.pooling_state_dict,
                "head_state_dict": self.head_state_dict,
                "provenance": self.provenance,
            },
            buffer,
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "EmotionCheckpointV2":
        buffer = io.BytesIO(data)
        payload = torch.load(buffer, map_location="cpu", weights_only=False)
        return cls(
            config=EmotionClassifierV2Config(**payload["config"]),
            backbone_state_dict=payload["backbone_state_dict"],
            pooling_state_dict=payload["pooling_state_dict"],
            head_state_dict=payload["head_state_dict"],
            provenance=payload.get("provenance", {}),
        )

    def build_model(self) -> Wav2VecAttentionEmotionNet:
        model = Wav2VecAttentionEmotionNet(self.config)
        model.backbone.load_state_dict(self.backbone_state_dict)
        model.pooling.load_state_dict(self.pooling_state_dict)
        model.head.load_state_dict(self.head_state_dict)
        model.eval()
        return model

    @classmethod
    def from_model(
        cls, model: Wav2VecAttentionEmotionNet, *, provenance: dict | None = None
    ) -> "EmotionCheckpointV2":
        return cls(
            config=model.config,
            backbone_state_dict=model.backbone.state_dict(),
            pooling_state_dict=model.pooling.state_dict(),
            head_state_dict=model.head.state_dict(),
            provenance=provenance or {},
        )
