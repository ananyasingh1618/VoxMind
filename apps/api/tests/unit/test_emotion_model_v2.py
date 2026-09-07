"""Unit tests for Emotion Model v2's architecture (services/emotion/v2/model.py).
Unlike v1's test_emotion_model.py (pure tensor-shape tests, no Hugging Face
dependency at all), these need `Wav2Vec2Config.from_pretrained` to read the
real facebook/wav2vec2-base architecture config - already cached locally in
this environment by Phase 3's real-model tests (a small JSON file, not the
full model weights), so no network access or the `real_model` marker is
needed here. Weights are never downloaded in this file: `Wav2Vec2Model(config)`
is a fresh random init, exactly what training starts from before loading
the real pretrained weights (train_emotion_model_v2.py does that step
separately, once, and is the only place that happens).
"""
from __future__ import annotations

import pytest
import torch

from voxmind.services.emotion.v2.model import (
    AttentionPooling,
    EmotionCheckpointV2,
    EmotionClassifierV2Config,
    Wav2VecAttentionEmotionNet,
)

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
BASE_MODEL = "facebook/wav2vec2-base"


def test_attention_pooling_ignores_padded_frames():
    pooling = AttentionPooling(hidden_dim=4)
    torch.manual_seed(0)
    hidden_states = torch.randn(1, 5, 4)
    # Real frames: first 3; padding: last 2.
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    pooled_with_mask = pooling(hidden_states, attention_mask)

    # Zeroing out (or changing) the padded frames must not change the
    # pooled output at all, since they should receive exactly zero weight.
    corrupted = hidden_states.clone()
    corrupted[:, 3:, :] = 999.0
    pooled_with_corrupted_padding = pooling(corrupted, attention_mask)

    assert torch.allclose(pooled_with_mask, pooled_with_corrupted_padding, atol=1e-5)


def test_attention_pooling_output_is_a_convex_combination_of_real_frames():
    pooling = AttentionPooling(hidden_dim=3)
    hidden_states = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

    pooled = pooling(hidden_states, attention_mask=None)

    # Weights sum to 1 (softmax) and are all non-negative, so each output
    # dimension must lie within the min/max of the real frames' values.
    assert pooled.shape == (1, 3)
    assert torch.all(pooled >= -1e-5) and torch.all(pooled <= 1.0 + 1e-5)


@pytest.fixture(scope="module")
def tiny_config() -> EmotionClassifierV2Config:
    from transformers import Wav2Vec2Config

    hidden_size = Wav2Vec2Config.from_pretrained(BASE_MODEL).hidden_size
    return EmotionClassifierV2Config(base_model=BASE_MODEL, backbone_hidden_dim=hidden_size, label_names=LABELS)


def test_config_num_classes_matches_label_count(tiny_config):
    assert tiny_config.num_classes == 6


def test_config_rejects_mismatched_hidden_dim():
    with pytest.raises(ValueError, match="does not match"):
        Wav2VecAttentionEmotionNet(
            EmotionClassifierV2Config(base_model=BASE_MODEL, backbone_hidden_dim=1, label_names=LABELS)
        )


@pytest.fixture(scope="module")
def small_model(tiny_config: EmotionClassifierV2Config) -> Wav2VecAttentionEmotionNet:
    return Wav2VecAttentionEmotionNet(tiny_config)


def test_forward_pass_produces_correct_logit_shape(small_model):
    small_model.eval()
    input_values = torch.randn(2, 4000)  # 2 clips, 0.25s @ 16kHz

    with torch.no_grad():
        logits = small_model(input_values)

    assert logits.shape == (2, len(LABELS))


def test_feature_extractor_is_always_frozen(small_model):
    assert all(not p.requires_grad for p in small_model.backbone.feature_extractor.parameters())


def test_freeze_backbone_freezes_every_backbone_parameter(small_model):
    small_model.unfreeze_top_layers(1)  # start from an unfrozen state
    small_model.freeze_backbone()

    assert all(not p.requires_grad for p in small_model.backbone.parameters())
    # Pooling/head are independent of backbone freezing.
    for p in small_model.pooling.parameters():
        p.requires_grad_(True)
    for p in small_model.head.parameters():
        p.requires_grad_(True)


def test_unfreeze_top_layers_only_unfreezes_the_requested_top_layers(small_model):
    small_model.freeze_backbone()
    n_layers = len(small_model.backbone.encoder.layers)

    small_model.unfreeze_top_layers(1)

    trainable_layer_indices = [
        i for i, layer in enumerate(small_model.backbone.encoder.layers)
        if any(p.requires_grad for p in layer.parameters())
    ]
    assert trainable_layer_indices == [n_layers - 1]
    # Feature extractor must remain frozen regardless.
    assert all(not p.requires_grad for p in small_model.backbone.feature_extractor.parameters())


def test_unfreeze_top_layers_rejects_out_of_range_count(small_model):
    n_layers = len(small_model.backbone.encoder.layers)
    with pytest.raises(ValueError):
        small_model.unfreeze_top_layers(n_layers + 1)
    with pytest.raises(ValueError):
        small_model.unfreeze_top_layers(0)


def test_checkpoint_round_trip_preserves_identical_predictions(small_model):
    small_model.eval()
    sample_input = torch.randn(1, 4000)

    with torch.no_grad():
        original_output = small_model(sample_input)

    checkpoint = EmotionCheckpointV2.from_model(small_model, provenance={"seed": 42})
    serialized = checkpoint.to_bytes()
    restored_checkpoint = EmotionCheckpointV2.from_bytes(serialized)
    restored_model = restored_checkpoint.build_model()

    with torch.no_grad():
        restored_output = restored_model(sample_input)

    assert torch.allclose(original_output, restored_output, atol=1e-5)
    assert restored_checkpoint.config.label_names == LABELS
    assert restored_checkpoint.provenance == {"seed": 42}
