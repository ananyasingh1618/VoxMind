"""Genuine real-model integration test: downloads and runs the actual
`emotion2vec/emotion2vec_plus_base` checkpoint via FunASR, extracts a real
768-dim utterance embedding from real audio, and runs it through a real
(untrained) `Emotion2VecClassifier` head - the same `real_model`-marked,
opt-in pattern already established for Whisper/Wav2Vec2 elsewhere in this
project (slow, no credentials needed, real inference, not run by default).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from voxmind.core.config import get_settings
from voxmind.services.emotion.emotion2vec.classifier_provider import Emotion2VecClassifier
from voxmind.services.emotion.emotion2vec.encoder import EMBEDDING_DIM, Emotion2VecEncoder
from voxmind.services.emotion.emotion2vec.model import Emotion2VecCheckpoint, Emotion2VecHead, Emotion2VecHeadConfig

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_real_encoder_produces_a_genuine_768_dim_embedding():
    settings = get_settings()
    encoder = Emotion2VecEncoder(settings)
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()

    embedding = await encoder.embed(wav_bytes, sample_rate=16000)

    assert embedding.dim == EMBEDDING_DIM
    assert len(embedding.vector) == EMBEDDING_DIM
    assert embedding.pooling == "emotion2vec-utterance"
    assert embedding.model_id == settings.EMOTION2VEC_MODEL
    # A genuine forward pass produces real, non-degenerate variance - not
    # an all-zeros or all-identical placeholder vector.
    assert len({round(v, 6) for v in embedding.vector}) > 10


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_real_embedding_flows_through_a_real_classifier_head():
    settings = get_settings()
    encoder = Emotion2VecEncoder(settings)
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()
    embedding = await encoder.embed(wav_bytes, sample_rate=16000)

    config = Emotion2VecHeadConfig(embedding_dim=EMBEDDING_DIM, label_names=LABELS, hidden_dim=32)
    model = Emotion2VecHead(config)
    checkpoint = Emotion2VecCheckpoint(config=config, state_dict=model.state_dict())
    classifier = Emotion2VecClassifier.from_checkpoint(checkpoint, model_version_id="test-real", trained=True)

    prediction = await classifier.predict(features=object(), embedding=embedding)  # type: ignore[arg-type]

    assert prediction.label in LABELS
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-4


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_real_encoder_is_deterministic_across_two_calls():
    """A frozen encoder given the same real audio twice must produce the
    same embedding - real determinism, not assumed."""
    settings = get_settings()
    encoder = Emotion2VecEncoder(settings)
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()

    e1 = await encoder.embed(wav_bytes, sample_rate=16000)
    e2 = await encoder.embed(wav_bytes, sample_rate=16000)

    assert e1.vector == e2.vector
