"""Real Wav2Vec2 speech embeddings via Hugging Face Transformers.

Pooling strategy (documented per project requirement): the model's frame-
level `last_hidden_state` (shape `[batch, time, 768]` for wav2vec2-base) is
reduced to a single fixed-size vector per clip via **mean pooling** over the
time dimension - the arithmetic mean of the hidden state across all frames.
This is the standard, well-established way to get an utterance-level
embedding from a frame-level self-supervised speech model when no
task-specific pooling head has been trained; it is *not* the model's
classification head (wav2vec2-base has no classification head at all - it's
the base self-supervised encoder), and it is not the CLS-token trick from
text transformers (Wav2Vec2 has no CLS token).

Each clip is processed one at a time (no batching), so there is no padding
and therefore no need for a padding-aware *masked* mean - a plain mean over
the real (unpadded) time axis is already exactly correct. Batched inference
with real attention-mask-aware pooling is a documented future optimization
if inference throughput ever requires it, not attempted here to avoid
depending on transformers-internal, version-fragile helper methods for a
case (batch size 1) where they aren't needed.

No `torch.Tensor` or Hugging Face model object crosses this class's public
`embed()` boundary - only the plain `SpeechEmbedding` from interfaces.py.
"""
from __future__ import annotations

import asyncio
import io

import numpy as np
import soundfile as sf
import structlog
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from voxmind.core.config import Settings
from voxmind.core.exceptions import AudioProcessingError
from voxmind.services.emotion.interfaces import SpeechEmbedding

logger = structlog.get_logger(__name__)


class HuggingFaceWav2Vec2Provider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Wav2Vec2Model | None = None
        self._feature_extractor: Wav2Vec2FeatureExtractor | None = None

    def _load(self) -> tuple[Wav2Vec2FeatureExtractor, Wav2Vec2Model]:
        if self._model is None or self._feature_extractor is None:
            try:
                self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    self._settings.WAV2VEC2_MODEL
                )
                model = Wav2Vec2Model.from_pretrained(self._settings.WAV2VEC2_MODEL)
                model.eval()
                model.to(self._settings.WAV2VEC2_DEVICE)
                self._model = model
            except Exception as exc:  # noqa: BLE001 - huggingface_hub raises many exception types
                logger.warning("wav2vec2_model_load_failed", model=self._settings.WAV2VEC2_MODEL)
                raise AudioProcessingError(
                    "Speech embedding model could not be loaded."
                ) from exc
        return self._feature_extractor, self._model

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
            logger.warning("wav2vec2_inference_failed", error=str(exc))
            raise AudioProcessingError("Speech embedding extraction failed during inference.") from exc

        return SpeechEmbedding(
            vector=vector.tolist(),
            dim=len(vector),
            model_id=self._settings.WAV2VEC2_MODEL,
            pooling=self._settings.WAV2VEC2_POOLING,
        )

    def _run_model(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        feature_extractor, model = self._load()

        # One clip, no batching, no padding - see module docstring for why a
        # plain (not masked) mean is exactly correct in this case.
        inputs = feature_extractor(samples, sampling_rate=sample_rate, return_tensors="pt")
        input_values = inputs["input_values"].to(self._settings.WAV2VEC2_DEVICE)

        with torch.no_grad():
            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state  # [1, T, 768]

        pooled = hidden_states.mean(dim=1)
        return pooled.squeeze(0).cpu().numpy()
