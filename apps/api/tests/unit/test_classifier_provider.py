from __future__ import annotations

import pytest

from voxmind.services.emotion.classifier_provider import (
    UNTRAINED_BASELINE_VERSION_ID,
    build_untrained_baseline,
)
from voxmind.services.emotion.interfaces import ACOUSTIC_VECTOR_DIM, AcousticFeatures, AcousticStat, SpeechEmbedding

LABELS = ["neutral", "happy", "sad"]


def _zero_stat() -> AcousticStat:
    return AcousticStat(mean=0.0, std=0.0, min=0.0, max=0.0, p25=0.0, p50=0.0, p75=0.0)


def _dummy_features() -> AcousticFeatures:
    return AcousticFeatures(
        duration_seconds=2.0,
        voiced_fraction=0.5,
        pitch_hz=_zero_stat(),
        rms_energy=_zero_stat(),
        zero_crossing_rate=_zero_stat(),
        spectral_centroid=_zero_stat(),
        spectral_bandwidth=_zero_stat(),
        spectral_rolloff=_zero_stat(),
        mfcc=[_zero_stat() for _ in range(13)],
    )


def _dummy_embedding(dim: int = 768) -> SpeechEmbedding:
    return SpeechEmbedding(vector=[0.0] * dim, dim=dim, model_id="test-model", pooling="mean")


@pytest.mark.asyncio
async def test_untrained_baseline_is_explicitly_marked_untrained():
    classifier = build_untrained_baseline(
        acoustic_dim=ACOUSTIC_VECTOR_DIM, embedding_dim=768, label_names=LABELS
    )

    prediction = await classifier.predict(_dummy_features(), _dummy_embedding())

    assert prediction.trained is False
    assert prediction.model_version_id == UNTRAINED_BASELINE_VERSION_ID
    assert prediction.label in LABELS


@pytest.mark.asyncio
async def test_probabilities_form_a_valid_distribution():
    classifier = build_untrained_baseline(
        acoustic_dim=ACOUSTIC_VECTOR_DIM, embedding_dim=768, label_names=LABELS
    )

    prediction = await classifier.predict(_dummy_features(), _dummy_embedding())

    assert set(prediction.probabilities.keys()) == set(LABELS)
    assert all(0.0 <= p <= 1.0 for p in prediction.probabilities.values())
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-5


@pytest.mark.asyncio
async def test_predicted_label_is_the_argmax_of_probabilities():
    classifier = build_untrained_baseline(
        acoustic_dim=ACOUSTIC_VECTOR_DIM, embedding_dim=768, label_names=LABELS
    )

    prediction = await classifier.predict(_dummy_features(), _dummy_embedding())

    assert prediction.label == max(prediction.probabilities, key=lambda k: prediction.probabilities[k])
