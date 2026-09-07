"""Real speaker diarization via pyannote.audio.

`pyannote/speaker-diarization-3.1` (and its `pyannote/segmentation-3.0`
dependency) are gated models on Hugging Face: using them for the first time
requires a free Hugging Face account that has accepted both models' license
terms, plus a read-scoped access token set as `HUGGINGFACE_TOKEN`. See
docs/audio.md for the exact steps. Without a valid token, this provider
raises a clear `AudioProcessingError` rather than attempting a request that
would fail with a confusing 401 deep inside pyannote/huggingface_hub.

No `pyannote.audio.Pipeline` or `pyannote.core.Annotation` object is ever
returned from this module - only the plain `DiarizationResult`/
`SpeakerSegment` models from `interfaces.py`.
"""
from __future__ import annotations

import asyncio
import io

import numpy as np
import soundfile as sf
import structlog
import torch
from pyannote.audio import Pipeline

from voxmind.core.config import Settings
from voxmind.core.exceptions import AudioProcessingError
from voxmind.services.speech.interfaces import DiarizationResult, SpeakerSegment

logger = structlog.get_logger(__name__)


class PyannoteDiarizationProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline: Pipeline | None = None

    def _get_pipeline(self) -> Pipeline:
        if self._pipeline is not None:
            return self._pipeline

        if not self._settings.HUGGINGFACE_TOKEN:
            raise AudioProcessingError(
                "Speaker diarization is not configured: HUGGINGFACE_TOKEN is required "
                "to download the gated pyannote model. See docs/audio.md."
            )
        try:
            # `token=` is this installed pyannote.audio version's real
            # parameter name (verified directly against
            # `inspect.signature(Pipeline.from_pretrained)` - `checkpoint,
            # revision, hparams_file, subfolder, token, cache_dir`, no
            # `use_auth_token` at all). The previous `use_auth_token=` kwarg
            # was written against an older pyannote.audio/huggingface_hub
            # convention that this project's `>=3.1,<5.0` pin allows to
            # drift past - it raised a real `TypeError: ... unexpected
            # keyword argument 'use_auth_token'` on every call once 4.x was
            # installed, which the broad `except Exception` below silently
            # reclassified as "invalid token/terms not accepted", masking
            # the actual cause. Found and fixed during real end-to-end
            # diarization verification once a genuine token was configured -
            # see docs/DECISIONS/0012. No dependency version was changed.
            self._pipeline = Pipeline.from_pretrained(
                self._settings.DIARIZATION_MODEL,
                token=self._settings.HUGGINGFACE_TOKEN,
            )
        except Exception as exc:  # noqa: BLE001 - huggingface_hub/pyannote raise many exception types
            # Never surface the raw exception (it can include the auth token
            # in a request-repr) - log a safe summary, raise a clean error.
            logger.warning("diarization_model_load_failed", model=self._settings.DIARIZATION_MODEL)
            raise AudioProcessingError(
                "Speaker diarization model could not be loaded. This usually means the "
                "Hugging Face token is invalid, or the gated model terms haven't been "
                "accepted for this account. See docs/audio.md."
            ) from exc

        if self._pipeline is None:
            raise AudioProcessingError(
                "Speaker diarization model failed to initialize (from_pretrained returned None)."
            )
        return self._pipeline

    async def diarize(self, audio_bytes: bytes, *, sample_rate: int) -> DiarizationResult:
        # Configuration is checked before touching the audio at all: a
        # missing token is a deployment/config problem, not something the
        # caller's audio could ever be at fault for, and should fail fast
        # regardless of whether the audio itself happens to be valid.
        if self._pipeline is None and not self._settings.HUGGINGFACE_TOKEN:
            raise AudioProcessingError(
                "Speaker diarization is not configured: HUGGINGFACE_TOKEN is required "
                "to download the gated pyannote model. See docs/audio.md."
            )

        try:
            samples, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:  # noqa: BLE001
            raise AudioProcessingError("Diarization received unreadable audio data.") from exc

        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        try:
            speaker_segments = await asyncio.to_thread(self._run_pipeline, samples, sample_rate)
        except AudioProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("diarization_inference_failed", error=str(exc))
            raise AudioProcessingError("Speaker diarization failed during inference.") from exc

        speaker_labels = sorted({s.speaker_label for s in speaker_segments})
        return DiarizationResult(
            speaker_segments=speaker_segments,
            num_speakers=len(speaker_labels),
            model_version=self._settings.DIARIZATION_MODEL,
        )

    def _run_pipeline(self, samples: np.ndarray, sample_rate: int) -> list[SpeakerSegment]:
        pipeline = self._get_pipeline()
        waveform = torch.from_numpy(samples).float().unsqueeze(0)  # shape (1, num_samples)
        result = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # This installed pyannote.audio version wraps the result in a
        # `DiarizeOutput` dataclass (`speaker_diarization`,
        # `exclusive_speaker_diarization`, `speaker_embeddings`) instead of
        # returning a plain `pyannote.core.Annotation` directly - a real
        # runtime shape change discovered live, once a genuine token+access
        # let inference actually run for the first time (see
        # docs/DECISIONS/0013). `exclusive_speaker_diarization` is pyannote's
        # own overlap-free variant, documented by the library itself as
        # "adapted to downstream transcription" - exactly this project's use
        # case (assigning each Whisper segment to one speaker in
        # `speech/alignment.py`). An older pyannote.audio 3.x pipeline
        # (still permitted by this project's `>=3.1,<5.0` pin) returns a
        # plain `Annotation` with no such attribute - handled below so both
        # continue to work without depending on which one is installed.
        annotation = getattr(result, "exclusive_speaker_diarization", result)

        # pyannote's raw labels ("SPEAKER_00", "SPEAKER_01", ...) are real
        # model output - normalized here into a stable internal convention
        # ("speaker_0", "speaker_1", ...) by first-appearance order, not
        # fabricated. The mapping is deterministic per call.
        label_order: dict[str, str] = {}
        segments: list[SpeakerSegment] = []
        # pyannote's Pipeline.__call__ return type is annotated as a union
        # that mypy can't narrow to Annotation here - it always is one at
        # runtime for this pipeline.
        for turn, _track, raw_label in annotation.itertracks(yield_label=True):  # type: ignore[union-attr]
            if raw_label not in label_order:
                label_order[raw_label] = f"speaker_{len(label_order)}"
            segments.append(
                SpeakerSegment(
                    speaker_label=label_order[raw_label],
                    start_ms=round(turn.start * 1000),
                    end_ms=round(turn.end * 1000),
                    confidence=None,  # pyannote's community pipeline doesn't expose this
                )
            )
        return segments
