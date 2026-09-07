"""Inference-time integration of Emotion Model v2 with VoxMind's *existing*,
unmodified pipeline abstractions (`../interfaces.py`'s
`Wav2Vec2EmbeddingProvider`/`EmotionClassifier` Protocols) and the
*existing* `EmotionService` orchestration (`../../emotion_service.py`) -
see docs/emotion.md's "Integration" section for the full reasoning; this
docstring covers only the one non-obvious design decision.

**Why the split is exactly at the embedding/classifier boundary**:
`EmotionService.process_message` dispatches two separate pipeline stages
per speaker turn - `AcousticFeatureExtractionStage` then
`Wav2Vec2EmbeddingStage` - and only then calls
`classifier.predict(features, embedding)` with their *already-computed,
already-JSON-serializable* outputs (see `../interfaces.py`: "no
torch.Tensor ... crosses this boundary"). By the time `predict()` runs, the
raw audio is gone - only a plain `AcousticFeatures` and a plain, already-
pooled `SpeechEmbedding.vector` remain. v2's attention pooling needs
frame-level Wav2Vec2 hidden states, which only exist *before* that
boundary. So v2's fine-tuned backbone + attention pooling must live inside
a `Wav2Vec2EmbeddingProvider` implementation (this module's
`AttentionWav2Vec2Provider`, replacing v1's fixed, frozen
`HuggingFaceWav2Vec2Provider` for this one call), producing a final pooled
768-dim vector exactly like v1's provider does (just via attention-pooling
its own fine-tuned backbone instead of mean-pooling a frozen one) -
`pooling="attention"` on the returned `SpeechEmbedding` is the only
observable difference. `TorchEmotionClassifierV2` then only needs the
small head (Linear->ReLU->Dropout->Linear) and, matching the Protocol
signature exactly, accepts `features: AcousticFeatures` but never reads it
- v2's architecture (docs/emotion.md, "Architecture") has no handcrafted-
acoustic-feature input at all, unlike v1's fused [acoustic; embedding]
design.

Net effect: `EmotionService`'s orchestration code is unchanged (same two
stage dispatches, same `classifier.predict(features, embedding)` call);
only *which concrete provider/classifier instances* are used for a given
call changes, resolved dynamically per active model version - see
`../active_classifier.py`. `AcousticFeatureExtractionStage` still runs and
its output is still computed for a v2-served request (discarded, unused) -
a deliberate, documented trade-off in favor of not touching
`EmotionService`'s dispatch sequence itself; see docs/emotion.md.
"""
from __future__ import annotations

import asyncio
import io

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor

from voxmind.core.config import Settings
from voxmind.core.exceptions import AudioProcessingError
from voxmind.services.emotion.interfaces import (
    AcousticFeatures,
    EmotionPrediction,
    SpeechEmbedding,
)
from voxmind.services.emotion.v2.model import EmotionCheckpointV2, EmotionHead, Wav2VecAttentionEmotionNet

POOLING_NAME = "attention"


class AttentionWav2Vec2Provider:
    """Implements the *existing*, unmodified `Wav2Vec2EmbeddingProvider`
    Protocol (../interfaces.py) using v2's fine-tuned backbone + learned
    attention pooling instead of v1's frozen backbone + mean pooling. Always
    runs in eval/no_grad mode - inference never fine-tunes further."""

    def __init__(self, model: Wav2VecAttentionEmotionNet, *, settings: Settings) -> None:
        self._model = model
        self._model.eval()
        self._settings = settings
        self._feature_extractor: Wav2Vec2FeatureExtractor | None = None

    def _load_feature_extractor(self) -> Wav2Vec2FeatureExtractor:
        # Same feature extractor class v1 uses (wav2vec_provider.py) - it's
        # a fixed, non-trainable normalization step (zero-mean/unit-var
        # waveform normalization + framing config), not a trained
        # component, so reusing it here duplicates zero learned state.
        if self._feature_extractor is None:
            self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self._model.config.base_model)
        return self._feature_extractor

    async def embed(self, audio_bytes: bytes, *, sample_rate: int) -> SpeechEmbedding:
        try:
            samples, actual_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:  # noqa: BLE001
            raise AudioProcessingError("Embedding extraction received unreadable audio.") from exc

        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            raise AudioProcessingError("Embedding extraction received empty audio.")

        try:
            vector = await asyncio.to_thread(self._run_model, samples, actual_rate)
        except AudioProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AudioProcessingError("Speech embedding extraction failed during inference.") from exc

        return SpeechEmbedding(
            vector=vector.tolist(), dim=len(vector), model_id=self._model.config.base_model, pooling=POOLING_NAME
        )

    def _run_model(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        feature_extractor = self._load_feature_extractor()
        # One clip, no padding needed (batch size 1, matching v1's
        # provider) - AttentionPooling's mask-aware path is exercised at
        # *training* time (real mini-batches, see
        # ml/training/train_emotion_model_v2.py); here attention_mask=None
        # is exactly correct, mirroring v1's documented same reasoning for
        # plain mean pooling in wav2vec_provider.py.
        inputs = feature_extractor(samples, sampling_rate=sample_rate, return_tensors="pt")
        input_values = inputs["input_values"]

        with torch.no_grad():
            outputs = self._model.backbone(input_values)
            hidden_states = outputs.last_hidden_state
            pooled = self._model.pooling(hidden_states, attention_mask=None)

        return pooled.squeeze(0).cpu().numpy()


class TorchEmotionClassifierV2:
    """Implements the *existing*, unmodified `EmotionClassifier` Protocol.
    Deliberately ignores `features` - see module docstring."""

    def __init__(
        self, *, head: EmotionHead, label_names: list[str], model_version_id: str, trained: bool
    ) -> None:
        self._head = head
        self._head.eval()
        self._label_names = label_names
        self._model_version_id = model_version_id
        self._trained = trained

    async def predict(self, features: AcousticFeatures, embedding: SpeechEmbedding) -> EmotionPrediction:
        del features  # v2's architecture has no handcrafted-acoustic-feature input - see module docstring.
        vector = torch.tensor(embedding.vector, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = self._head(vector)
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


def build_inference_components(
    checkpoint: EmotionCheckpointV2, *, model_version_id: str, trained: bool, settings: Settings
) -> tuple[AttentionWav2Vec2Provider, TorchEmotionClassifierV2]:
    """Splits one trained `EmotionCheckpointV2` into the two Protocol-
    shaped objects `../active_classifier.py` needs - loaded and
    reconstructed exactly once per call, never duplicated."""
    model = checkpoint.build_model()
    provider = AttentionWav2Vec2Provider(model, settings=settings)
    classifier = TorchEmotionClassifierV2(
        head=model.head,
        label_names=checkpoint.config.label_names,
        model_version_id=model_version_id,
        trained=trained,
    )
    return provider, classifier
