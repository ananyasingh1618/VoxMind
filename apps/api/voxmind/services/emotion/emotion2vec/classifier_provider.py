"""Inference-time wrapper around a trained `Emotion2VecHead`, implementing
the existing, unmodified `EmotionClassifier` Protocol (../interfaces.py) -
mirrors v1's `classifier_provider.py` and v2's `v2/inference.py`
exactly. Deliberately ignores `features: AcousticFeatures` (the Protocol's
required parameter) for the same reason v2 does: this architecture has no
handcrafted-acoustic-feature input, only the emotion2vec+ embedding.
"""
from __future__ import annotations

import torch

from voxmind.services.emotion.emotion2vec.model import Emotion2VecCheckpoint, Emotion2VecHead
from voxmind.services.emotion.interfaces import AcousticFeatures, EmotionPrediction, SpeechEmbedding


class Emotion2VecClassifier:
    def __init__(self, *, head: Emotion2VecHead, label_names: list[str], model_version_id: str, trained: bool) -> None:
        self._head = head
        self._head.eval()
        self._label_names = label_names
        self._model_version_id = model_version_id
        self._trained = trained

    @classmethod
    def from_checkpoint(cls, checkpoint: Emotion2VecCheckpoint, *, model_version_id: str, trained: bool) -> "Emotion2VecClassifier":
        return cls(
            head=checkpoint.build_model(),
            label_names=checkpoint.config.label_names,
            model_version_id=model_version_id,
            trained=trained,
        )

    async def predict(self, features: AcousticFeatures, embedding: SpeechEmbedding) -> EmotionPrediction:
        del features  # v2/emotion2vec architectures have no handcrafted-acoustic-feature input - see module docstring.
        vector = torch.tensor(embedding.vector, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self._head(vector)
            probabilities_tensor = torch.softmax(logits, dim=-1).squeeze(0)

        probabilities = {label: float(prob) for label, prob in zip(self._label_names, probabilities_tensor)}
        predicted_label = max(probabilities, key=lambda label: probabilities[label])

        return EmotionPrediction(
            label=predicted_label,
            probabilities=probabilities,
            model_version_id=self._model_version_id,
            trained=self._trained,
        )
