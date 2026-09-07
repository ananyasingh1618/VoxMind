from __future__ import annotations

import torch

from voxmind.services.emotion.model import (
    EmotionCheckpoint,
    EmotionClassifierConfig,
    EmotionClassifierNet,
    FeatureNormalizer,
)

LABELS = ["neutral", "happy", "sad", "angry"]


def test_classifier_config_derived_properties():
    config = EmotionClassifierConfig(acoustic_dim=134, embedding_dim=768, label_names=LABELS)
    assert config.num_classes == 4
    assert config.input_dim == 134 + 768


def test_forward_pass_produces_correct_logit_shape():
    config = EmotionClassifierConfig(acoustic_dim=10, embedding_dim=6, label_names=LABELS, hidden_dim=8)
    model = EmotionClassifierNet(config)
    batch = torch.randn(3, config.input_dim)

    logits = model(batch)

    assert logits.shape == (3, len(LABELS))


def test_normalizer_fit_produces_zero_mean_unit_std_on_training_data():
    vectors = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
    normalizer = FeatureNormalizer.fit(vectors)

    transformed = torch.stack([normalizer.transform(v) for v in vectors])

    assert torch.allclose(transformed.mean(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(transformed.std(dim=0, unbiased=False), torch.ones(2), atol=1e-5)


def test_normalizer_handles_constant_dimension_without_dividing_by_zero():
    vectors = [[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]  # first dim is constant
    normalizer = FeatureNormalizer.fit(vectors)

    result = normalizer.transform([5.0, 2.0])

    assert torch.isfinite(result).all()
    assert result[0].item() == 0.0  # constant dim maps to 0, not NaN/inf


def test_checkpoint_round_trip_preserves_identical_predictions():
    config = EmotionClassifierConfig(acoustic_dim=4, embedding_dim=4, label_names=LABELS, hidden_dim=8)
    model = EmotionClassifierNet(config)
    model.eval()  # dropout must be disabled for a fair before/after comparison
    normalizer = FeatureNormalizer(mean=[0.0] * config.input_dim, std=[1.0] * config.input_dim)

    sample_input = torch.randn(1, config.input_dim)
    with torch.no_grad():
        original_output = model(sample_input)

    checkpoint = EmotionCheckpoint(config=config, normalizer=normalizer, state_dict=model.state_dict())
    serialized = checkpoint.to_bytes()
    restored = EmotionCheckpoint.from_bytes(serialized)
    restored_model = restored.build_model()

    with torch.no_grad():
        restored_output = restored_model(sample_input)

    assert torch.allclose(original_output, restored_output)
    assert restored.config.label_names == LABELS
    assert restored.normalizer.mean == normalizer.mean
