"""Speech service boundary (preprocessing, STT, diarization, alignment).

Every provider Protocol here is implemented for real in this package
(preprocessing.py, whisper_provider.py, diarization_provider.py,
alignment.py) - nothing in this module is a placeholder. What a given
deployment can actually *execute* depends on real external factors (model
downloads, Hugging Face gating/credentials for diarization) documented in
docs/audio.md - the code path is real either way; only whether inference can
complete in a given environment varies.
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class PreprocessedAudio(BaseModel):
    """Output of real audio decoding/normalization - never the raw upload.
    `storage_key` points at a canonical mono, 16kHz, 16-bit PCM WAV file
    written back to the storage abstraction, distinct from the original
    upload's key."""

    storage_key: str
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str = "wav"


class TranscriptSegment(BaseModel):
    start_ms: int
    end_ms: int
    text: str
    # A confidence *proxy* derived from the model's average log-probability
    # for this segment (exp(avg_logprob)), not a calibrated probability -
    # faster-whisper does not provide one. None if the provider genuinely
    # doesn't expose per-segment likelihood at all.
    confidence: float | None = None


class TranscriptionResult(BaseModel):
    text: str
    segments: list[TranscriptSegment]
    language: str
    language_probability: float | None = None
    model_version: str


class SpeakerSegment(BaseModel):
    speaker_label: str
    start_ms: int
    end_ms: int
    # pyannote's community diarization pipeline does not expose a
    # per-segment confidence score, so this is None today - present in the
    # schema (not omitted) so a future provider that does supply one doesn't
    # require a contract change.
    confidence: float | None = None


class DiarizationResult(BaseModel):
    speaker_segments: list[SpeakerSegment]
    num_speakers: int
    model_version: str


class AlignedTurn(BaseModel):
    """Derived output of the alignment stage - explicitly distinct from raw
    transcription/diarization output. See alignment.py for the algorithm."""

    start_ms: int
    end_ms: int
    text: str
    speaker_label: str | None = Field(
        default=None, description="None when no diarization segment overlapped this turn."
    )


class SpeechToTextProvider(Protocol):
    async def transcribe(
        self, audio_bytes: bytes, *, sample_rate: int, language: str | None = None
    ) -> TranscriptionResult: ...


class DiarizationProvider(Protocol):
    async def diarize(self, audio_bytes: bytes, *, sample_rate: int) -> DiarizationResult: ...
