"""Pure tensor-shape/round-trip tests for the emotion2vec+ head
(services/emotion/emotion2vec/model.py) - no real model download needed,
mirrors test_emotion_model.py's (v1) and test_emotion_model_v2.py's
existing pattern exactly."""
from __future__ import annotations

import torch

from voxmind.services.emotion.emotion2vec.model import Emotion2VecCheckpoint, Emotion2VecHead, Emotion2VecHeadConfig

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


def test_config_derived_properties():
    config = Emotion2VecHeadConfig(embedding_dim=768, label_names=LABELS)
    assert config.num_classes == 6


def test_forward_pass_produces_correct_logit_shape():
    config = Emotion2VecHeadConfig(embedding_dim=768, label_names=LABELS, hidden_dim=32)
    model = Emotion2VecHead(config)
    batch = torch.randn(4, 768)

    logits = model(batch)

    assert logits.shape == (4, len(LABELS))


def test_checkpoint_round_trip_preserves_identical_predictions():
    config = Emotion2VecHeadConfig(embedding_dim=768, label_names=LABELS, hidden_dim=32)
    model = Emotion2VecHead(config)
    model.eval()

    sample_input = torch.randn(1, 768)
    with torch.no_grad():
        original_output = model(sample_input)

    checkpoint = Emotion2VecCheckpoint(config=config, state_dict=model.state_dict(), provenance={"seed": 42})
    serialized = checkpoint.to_bytes()
    restored = Emotion2VecCheckpoint.from_bytes(serialized)
    restored_model = restored.build_model()

    with torch.no_grad():
        restored_output = restored_model(sample_input)

    assert torch.allclose(original_output, restored_output)
    assert restored.config.label_names == LABELS
    assert restored.provenance == {"seed": 42}


def test_restored_model_is_in_eval_mode():
    config = Emotion2VecHeadConfig(embedding_dim=768, label_names=LABELS)
    model = Emotion2VecHead(config)
    checkpoint = Emotion2VecCheckpoint(config=config, state_dict=model.state_dict())

    restored_model = checkpoint.build_model()

    assert restored_model.training is False
