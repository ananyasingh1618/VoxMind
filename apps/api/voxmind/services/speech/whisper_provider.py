"""Real speech-to-text via faster-whisper (CTranslate2-accelerated Whisper).

This is the only module that imports `faster_whisper` - no
`faster_whisper.WhisperModel` or `faster_whisper.transcribe.Segment` object
is ever returned from this module; everything crosses the boundary as the
plain `TranscriptionResult`/`TranscriptSegment` Pydantic models from
`interfaces.py`. Swapping to a hosted/API-backed Whisper provider later
means writing one new class satisfying `SpeechToTextProvider` - nothing
above this module changes.
"""
from __future__ import annotations

import asyncio
import io
import math

import numpy as np
import soundfile as sf
import structlog
from faster_whisper import WhisperModel

from voxmind.core.config import Settings
from voxmind.core.exceptions import AudioProcessingError
from voxmind.services.speech.interfaces import TranscriptionResult, TranscriptSegment

logger = structlog.get_logger(__name__)


def _load_model(settings: Settings) -> WhisperModel:
    return WhisperModel(
        settings.WHISPER_MODEL_SIZE,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
    )


def _logprob_to_confidence(avg_logprob: float | None) -> float | None:
    """faster-whisper exposes `avg_logprob` (average log-probability of the
    tokens in a segment), not a calibrated confidence score. exp() maps it
    back into a (0, 1] range that reads like a probability - useful as a
    rough proxy, but this is explicitly documented as a proxy, not a true
    calibrated confidence, both here and in TranscriptSegment's docstring.
    """
    if avg_logprob is None:
        return None
    return math.exp(avg_logprob)


class FasterWhisperProvider:
    """Real `SpeechToTextProvider` implementation. Model weights are
    downloaded (once, cached by huggingface_hub) from the
    `Systran/faster-whisper-<size>` repos on first use - these are NOT
    gated, so no Hugging Face token is required, unlike diarization.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            self._model = _load_model(self._settings)
        return self._model

    async def transcribe(
        self, audio_bytes: bytes, *, sample_rate: int, language: str | None = None
    ) -> TranscriptionResult:
        try:
            samples, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:  # noqa: BLE001
            raise AudioProcessingError("Whisper received unreadable audio data.") from exc

        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        try:
            segments, info = await asyncio.to_thread(
                self._run_model, samples, sample_rate, language
            )
        except Exception as exc:  # noqa: BLE001 - any faster-whisper/ctranslate2 failure
            logger.warning("whisper_transcription_failed", error=str(exc))
            raise AudioProcessingError("Speech-to-text transcription failed.") from exc

        transcript_segments = [
            TranscriptSegment(
                start_ms=round(seg.start * 1000),
                end_ms=round(seg.end * 1000),
                text=seg.text.strip(),
                confidence=_logprob_to_confidence(seg.avg_logprob),
            )
            for seg in segments
        ]
        full_text = " ".join(s.text for s in transcript_segments).strip()

        return TranscriptionResult(
            text=full_text,
            segments=transcript_segments,
            language=info.language,
            language_probability=info.language_probability,
            model_version=f"faster-whisper:{self._settings.WHISPER_MODEL_SIZE}",
        )

    def _run_model(self, samples: np.ndarray, sample_rate: int, language: str | None):
        model = self._get_model()
        # faster-whisper's WhisperModel.transcribe expects 16kHz audio; the
        # preprocessing stage guarantees this, but resample defensively if a
        # caller ever passes something else directly to this provider.
        if sample_rate != 16000:
            samples = _resample_linear(samples, sample_rate, 16000)
        segments_generator, info = model.transcribe(
            samples,
            language=language or self._settings.WHISPER_LANGUAGE,
            beam_size=self._settings.WHISPER_BEAM_SIZE,
        )
        # The generator is lazy - materialize it here, inside the worker
        # thread, so no faster-whisper internals leak into the async caller.
        return list(segments_generator), info


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    duration = len(samples) / source_rate
    target_length = int(duration * target_rate)
    source_indices = np.linspace(0, len(samples) - 1, num=target_length)
    return np.interp(source_indices, np.arange(len(samples)), samples).astype(np.float32)
