"""Pipeline stages implementing the approved `PipelineStage` contract
(`workers/task_runner.py`): each stage takes and returns a plain,
JSON-serializable Pydantic model, and knows nothing about HTTP, the
database, or the TaskRunner dispatching it. Framework/model-specific objects
(faster-whisper Segments, pyannote Annotations) never cross a stage
boundary - only the plain models from `interfaces.py`.

Every stage downloads whatever audio it needs from the storage abstraction
by key and stays otherwise stateless between calls, which is what makes
each one independently dispatchable once Celery is introduced.
"""
from __future__ import annotations

from pydantic import BaseModel

from voxmind.core.config import Settings
from voxmind.services.speech.alignment import align_transcript_with_speakers
from voxmind.services.speech.diarization_provider import PyannoteDiarizationProvider
from voxmind.services.speech.interfaces import (
    AlignedTurn,
    DiarizationResult,
    PreprocessedAudio,
    SpeakerSegment,
    TranscriptionResult,
    TranscriptSegment,
)
from voxmind.services.speech.preprocessing import preprocess_audio
from voxmind.services.speech.whisper_provider import FasterWhisperProvider
from voxmind.services.storage.interfaces import StorageBackend


class PreprocessInput(BaseModel):
    original_storage_key: str
    processed_storage_key: str


class PreprocessOutput(BaseModel):
    audio: PreprocessedAudio


class AudioPreprocessingStage:
    name = "audio_preprocessing"

    def __init__(self, storage: StorageBackend, settings: Settings) -> None:
        self._storage = storage
        self._settings = settings

    async def run(self, input: PreprocessInput) -> PreprocessOutput:
        raw_bytes = await self._storage.download(input.original_storage_key)
        metadata, wav_bytes = await preprocess_audio(raw_bytes, settings=self._settings)
        await self._storage.upload(input.processed_storage_key, wav_bytes, content_type="audio/wav")
        metadata.storage_key = input.processed_storage_key
        return PreprocessOutput(audio=metadata)


class TranscriptionInput(BaseModel):
    processed_storage_key: str
    sample_rate: int
    language: str | None = None


class TranscriptionOutput(BaseModel):
    transcription: TranscriptionResult


class WhisperTranscriptionStage:
    name = "whisper_transcription"

    def __init__(self, storage: StorageBackend, provider: FasterWhisperProvider) -> None:
        self._storage = storage
        self._provider = provider

    async def run(self, input: TranscriptionInput) -> TranscriptionOutput:
        audio_bytes = await self._storage.download(input.processed_storage_key)
        result = await self._provider.transcribe(
            audio_bytes, sample_rate=input.sample_rate, language=input.language
        )
        return TranscriptionOutput(transcription=result)


class DiarizationInput(BaseModel):
    processed_storage_key: str
    sample_rate: int


class DiarizationOutput(BaseModel):
    diarization: DiarizationResult


class SpeakerDiarizationStage:
    name = "speaker_diarization"

    def __init__(self, storage: StorageBackend, provider: PyannoteDiarizationProvider) -> None:
        self._storage = storage
        self._provider = provider

    async def run(self, input: DiarizationInput) -> DiarizationOutput:
        audio_bytes = await self._storage.download(input.processed_storage_key)
        result = await self._provider.diarize(audio_bytes, sample_rate=input.sample_rate)
        return DiarizationOutput(diarization=result)


class AlignmentInput(BaseModel):
    transcript_segments: list[TranscriptSegment]
    speaker_segments: list[SpeakerSegment]


class AlignmentOutput(BaseModel):
    turns: list[AlignedTurn]


class TranscriptAlignmentStage:
    """Pure/deterministic - no I/O, no ML. Still implemented as a
    PipelineStage so it dispatches through the same TaskRunner contract as
    every other stage, keeping the orchestrator uniform."""

    name = "transcript_alignment"

    async def run(self, input: AlignmentInput) -> AlignmentOutput:
        turns = align_transcript_with_speakers(input.transcript_segments, input.speaker_segments)
        return AlignmentOutput(turns=turns)
