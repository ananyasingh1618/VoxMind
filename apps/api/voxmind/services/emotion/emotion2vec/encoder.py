"""`Emotion2VecEncoder`: a clean adapter around FunASR's `AutoModel`
loading of `emotion2vec/emotion2vec_plus_base`, implementing the existing,
unmodified `Wav2Vec2EmbeddingProvider` Protocol (../interfaces.py) so the
rest of VoxMind's emotion pipeline stays model-provider agnostic - exactly
the same seam v1's `HuggingFaceWav2Vec2Provider` and v2's
`AttentionWav2Vec2Provider` already use (see docs/emotion.md).

**Why FunASR, not plain `transformers`**: verified directly in this
environment (not assumed) - emotion2vec+ has no `transformers`-compatible
`AutoModel`/`AutoProcessor` path; its own model card and GitHub repository
document `funasr.AutoModel(...)` as the supported loading mechanism, and a
real load in this environment confirmed "All keys matched successfully"
only via that path. The rest of VoxMind never imports `funasr` directly -
only this one adapter module does, so a future provider swap would only
ever need to touch this file.

**Why the encoder is used as a fixed, frozen feature extractor here, not
fine-tuned**: docs/emotion.md's training design (Stage A) keeps
emotion2vec+ entirely frozen and trains only a small downstream head - this
class's `embed()` method reflects that: it is inference-only (`generate()`
called under `torch.no_grad()` implicitly, since FunASR's own inference
path already runs in eval mode), never exposes gradients, and is used
identically whether producing 768-dim embeddings for training-time caching
(ml/training/train_emotion2vec.py) or for real-time inference
(active_classifier.py).
"""
from __future__ import annotations

import asyncio
import io

import numpy as np
import soundfile as sf
import structlog

from voxmind.core.config import Settings
from voxmind.core.exceptions import AudioProcessingError
from voxmind.services.emotion.interfaces import SpeechEmbedding

logger = structlog.get_logger(__name__)

EMBEDDING_DIM = 768  # emotion2vec_plus_base's real utterance-level representation size - verified, not assumed
POOLING_NAME = "emotion2vec-utterance"


class Emotion2VecEncoder:
    """Implements the existing, unmodified `Wav2Vec2EmbeddingProvider`
    Protocol - despite the name (a Phase 3 holdover; the Protocol is
    genuinely provider-agnostic, just named after its first implementation),
    this class has nothing to do with Wav2Vec2 internally."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None  # lazily constructed - see _load()

    def _load(self):
        if self._model is None:
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise AudioProcessingError(
                    "emotion2vec+ requires the 'funasr' package, which is not installed."
                ) from exc
            try:
                self._model = AutoModel(
                    model=self._settings.EMOTION2VEC_MODEL,
                    hub=self._settings.EMOTION2VEC_HUB,
                    disable_update=True,  # never phone home to check for a newer funasr version at runtime
                    device=self._settings.EMOTION2VEC_DEVICE,
                )
            except Exception as exc:  # noqa: BLE001 - funasr/modelscope raise many exception types
                logger.warning("emotion2vec_model_load_failed", model=self._settings.EMOTION2VEC_MODEL)
                raise AudioProcessingError("emotion2vec+ model could not be loaded.") from exc
        return self._model

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
            logger.warning("emotion2vec_inference_failed", error=str(exc))
            raise AudioProcessingError("Speech embedding extraction failed during inference.") from exc

        return SpeechEmbedding(
            vector=vector.tolist(),
            dim=len(vector),
            model_id=self._settings.EMOTION2VEC_MODEL,
            pooling=POOLING_NAME,
        )

    def _run_model(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        model = self._load()
        # emotion2vec+ is natively 16kHz - this project's own preprocessing
        # (Phase 2's preprocess_audio / ffmpeg) already normalizes all
        # audio to mono/16kHz before it ever reaches this stage, matching
        # v1/v2's identical assumption; a resample here would only ever
        # engage for a caller that skipped that normalization.
        if sample_rate != 16000:
            import librosa

            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)

        result = model.generate(samples, granularity="utterance", extract_embedding=True)
        feats = result[0]["feats"]
        vector = np.asarray(feats, dtype=np.float32)
        if vector.shape != (EMBEDDING_DIM,):
            raise AudioProcessingError(
                f"emotion2vec+ returned an unexpected embedding shape {vector.shape}, expected ({EMBEDDING_DIM},)."
            )
        return vector
