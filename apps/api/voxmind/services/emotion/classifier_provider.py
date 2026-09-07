"""Inference-time wrapper around a loaded `EmotionClassifierNet`.

This class only ever runs whatever model/label-mapping/normalizer it was
constructed with - it has no opinion on whether that model is fit for
production use. That decision (require a `trained=True`, active
`ModelVersion`) is made one layer up, in `services/emotion_service.py`,
which is the *only* code path reachable from the product-facing API. The
`build_untrained_baseline` factory here exists solely so tests can exercise
the real forward-pass/pipeline plumbing without a trained checkpoint - it is
never imported by `api/v1/endpoints/emotion.py`.
"""
from __future__ import annotations

import torch

from voxmind.services.emotion.interfaces import AcousticFeatures, EmotionPrediction, SpeechEmbedding
from voxmind.services.emotion.model import (
    EmotionCheckpoint,
    EmotionClassifierConfig,
    EmotionClassifierNet,
    FeatureNormalizer,
)

UNTRAINED_BASELINE_VERSION_ID = "untrained-baseline"


class TorchEmotionClassifier:
    def __init__(
        self,
        *,
        model: EmotionClassifierNet,
        normalizer: FeatureNormalizer,
        label_names: list[str],
        model_version_id: str,
        trained: bool,
    ) -> None:
        self._model = model
        self._model.eval()
        self._normalizer = normalizer
        self._label_names = label_names
        self._model_version_id = model_version_id
        self._trained = trained

    @classmethod
    def from_checkpoint(
        cls, checkpoint: EmotionCheckpoint, *, model_version_id: str, trained: bool
    ) -> "TorchEmotionClassifier":
        return cls(
            model=checkpoint.build_model(),
            normalizer=checkpoint.normalizer,
            label_names=checkpoint.config.label_names,
            model_version_id=model_version_id,
            trained=trained,
        )

    async def predict(
        self, features: AcousticFeatures, embedding: SpeechEmbedding
    ) -> EmotionPrediction:
        vector = features.to_vector() + embedding.vector
        normalized = self._normalizer.transform(vector).unsqueeze(0)

        with torch.no_grad():
            logits = self._model(normalized)
            probabilities_tensor = torch.softmax(logits, dim=-1).squeeze(0)

        probabilities = {
            label: float(prob) for label, prob in zip(self._label_names, probabilities_tensor)
        }
        predicted_label = max(probabilities, key=lambda label: probabilities[label])

        return EmotionPrediction(
            label=predicted_label,
            probabilities=probabilities,
            model_version_id=self._model_version_id,
            trained=self._trained,
        )


def build_untrained_baseline(
    *, acoustic_dim: int, embedding_dim: int, label_names: list[str]
) -> TorchEmotionClassifier:
    """Random-initialized network for pipeline/architecture testing only.
    Its output is real (genuinely computed) tensor math, but meaningless as
    an emotion signal - `trained=False` makes that explicit on every
    prediction it produces, and it is never reachable from the API."""
    config = EmotionClassifierConfig(
        acoustic_dim=acoustic_dim, embedding_dim=embedding_dim, label_names=label_names
    )
    model = EmotionClassifierNet(config)
    normalizer = FeatureNormalizer(mean=[0.0] * config.input_dim, std=[1.0] * config.input_dim)
    return TorchEmotionClassifier(
        model=model,
        normalizer=normalizer,
        label_names=label_names,
        model_version_id=UNTRAINED_BASELINE_VERSION_ID,
        trained=False,
    )
