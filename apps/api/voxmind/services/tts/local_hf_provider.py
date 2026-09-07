"""Real local text-to-speech via `facebook/mms-tts-eng` - a VITS model, not
gated, freely downloadable. Same pattern as every other local model in this
project (Wav2Vec2, MiniLM, the cross-encoder reranker): loaded once via
plain Hugging Face Transformers, run synchronously in a worker thread.

Output is a raw float32 waveform at the model's native sample rate (`VitsModel`
has no vocoder step - the model outputs audio samples directly), WAV-encoded
here via `soundfile` (already a dependency since Phase 2) so the result is a
playable file, not a bare tensor.
"""
from __future__ import annotations

import asyncio
import io

import numpy as np
import soundfile as sf
import structlog
import torch
from transformers import VitsModel, VitsTokenizer

from voxmind.core.config import Settings
from voxmind.core.exceptions import PipelineProcessingError
from voxmind.services.tts.interfaces import TtsResult

logger = structlog.get_logger(__name__)


class LocalHfTtsProvider:
    provider_name = "local_hf"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Typed loosely - see the matching comment in
        # services/knowledge/embedding_provider.py for why "Auto*"/specific
        # HF model classes aren't used as instance-attribute type hints here.
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None or self._tokenizer is None:
            try:
                self._tokenizer = VitsTokenizer.from_pretrained(self._settings.TTS_LOCAL_MODEL)
                model = VitsModel.from_pretrained(self._settings.TTS_LOCAL_MODEL)
                model.eval()
                model.to(self._settings.TTS_DEVICE)
                self._model = model
            except Exception as exc:  # noqa: BLE001
                logger.warning("tts_model_load_failed", model=self._settings.TTS_LOCAL_MODEL)
                raise PipelineProcessingError("Local TTS model could not be loaded.") from exc
        return self._tokenizer, self._model

    async def synthesize(self, text: str, voice_profile: str | None = None) -> TtsResult:
        if not text.strip():
            raise PipelineProcessingError("Cannot synthesize empty text.")
        try:
            return await asyncio.to_thread(self._run, text)
        except PipelineProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("tts_inference_failed", error=str(exc))
            raise PipelineProcessingError("Local TTS synthesis failed during inference.") from exc

    def _run(self, text: str) -> TtsResult:
        tokenizer, model = self._load()
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._settings.TTS_DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs).waveform

        samples = output.squeeze(0).cpu().numpy().astype(np.float32)
        sample_rate = model.config.sampling_rate

        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV")
        audio_bytes = buffer.getvalue()
        duration_ms = round(len(samples) / sample_rate * 1000)

        return TtsResult(
            audio_bytes=audio_bytes,
            content_type="audio/wav",
            provider=f"local_hf:{self._settings.TTS_LOCAL_MODEL}",
            duration_ms=duration_ms,
        )
