"""Tests Emotion Model v2's inference-time integration
(services/emotion/v2/inference.py) against a small, untrained (randomly
initialized) model - proves the real forward-pass/Protocol plumbing works,
exactly like v1's own untrained-baseline pattern
(classifier_provider.py::build_untrained_baseline) - not that predictions
are meaningful (they aren't, for a random network)."""
from __future__ import annotations

from pathlib import Path

import pytest

from voxmind.core.config import get_settings
from voxmind.services.emotion.v2.inference import build_inference_components
from voxmind.services.emotion.v2.model import EmotionCheckpointV2, EmotionClassifierV2Config, Wav2VecAttentionEmotionNet

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
BASE_MODEL = "facebook/wav2vec2-base"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def small_checkpoint() -> EmotionCheckpointV2:
    from transformers import Wav2Vec2Config

    hidden_size = Wav2Vec2Config.from_pretrained(BASE_MODEL).hidden_size
    config = EmotionClassifierV2Config(base_model=BASE_MODEL, backbone_hidden_dim=hidden_size, label_names=LABELS)
    model = Wav2VecAttentionEmotionNet(config)
    return EmotionCheckpointV2.from_model(model, provenance={"note": "untrained, test-only"})


@pytest.mark.asyncio
async def test_provider_and_classifier_together_produce_a_valid_prediction(small_checkpoint):
    settings = get_settings()
    provider, classifier = build_inference_components(
        small_checkpoint, model_version_id="test-v2-fixture", trained=True, settings=settings
    )

    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    embedding = await provider.embed(wav_bytes, sample_rate=16000)

    assert embedding.dim == small_checkpoint.config.backbone_hidden_dim
    assert embedding.pooling == "attention"
    assert embedding.model_id == BASE_MODEL

    # `features` is required by the Protocol signature but deliberately
    # unused by v2 - passing None-shaped garbage proves that (a real
    # AcousticFeatures is a large object; a call that touched it would
    # error on this input rather than silently ignoring it).
    prediction = await classifier.predict(features=object(), embedding=embedding)  # type: ignore[arg-type]

    assert prediction.label in LABELS
    assert prediction.trained is True
    assert prediction.model_version_id == "test-v2-fixture"
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-4
    assert set(prediction.probabilities.keys()) == set(LABELS)


@pytest.mark.asyncio
async def test_provider_rejects_empty_audio(small_checkpoint):
    from voxmind.core.exceptions import AudioProcessingError

    settings = get_settings()
    provider, _classifier = build_inference_components(
        small_checkpoint, model_version_id="test-v2-fixture", trained=True, settings=settings
    )

    with pytest.raises(AudioProcessingError):
        await provider.embed(b"", sample_rate=16000)
