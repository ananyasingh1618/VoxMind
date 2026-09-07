"""Audio application service: ingestion, and orchestration of the speech
pipeline through the approved TaskRunner contract
(`dispatch → get_status → get_result`, unchanged from Phase 1).

Error-handling note on why diarization failures don't fail the whole job:
`InProcessTaskRunner.dispatch()` (approved, unmodified) catches any
exception a stage raises and collapses it to a `JobHandle(status=FAILED,
error=str(exc))` - the original exception *type* does not survive that
boundary, only its message. That's fine for preprocessing/transcription
(a failure there is genuinely fatal - there's no transcript without them),
but diarization is a case where a very common, expected condition (no
`HUGGINGFACE_TOKEN` configured - see docs/audio.md) shouldn't destroy an
otherwise-successful transcript. Rather than fragile string-matching the
collapsed error message to tell "not configured" apart from "failed", this
service checks `settings.HUGGINGFACE_TOKEN` itself *before* deciding whether
to dispatch the diarization stage at all, and records
`diarization_status="unavailable"` directly - no dispatch attempted, no
information lost, no guessing.
"""
from __future__ import annotations

import time
import uuid
from typing import cast

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings
from voxmind.core.exceptions import AudioProcessingError, NotFoundError
from voxmind.models.audio_asset import AudioAsset
from voxmind.models.audio_processing_job import AudioProcessingJob
from voxmind.models.message import Message
from voxmind.repositories.audio_asset_repository import AudioAssetRepository
from voxmind.repositories.audio_processing_job_repository import AudioProcessingJobRepository
from voxmind.repositories.message_repository import MessageRepository
from voxmind.repositories.transcript_repository import TranscriptRepository
from voxmind.services.conversation_service import ConversationService
from voxmind.services.speech.audio_validation import safe_filename, validate_audio_upload
from voxmind.services.speech.diarization_provider import PyannoteDiarizationProvider
from voxmind.services.speech.interfaces import (
    AlignedTurn,
    DiarizationResult,
    PreprocessedAudio,
    TranscriptionResult,
)
from voxmind.services.speech.stages import (
    AlignmentInput,
    AlignmentOutput,
    AudioPreprocessingStage,
    DiarizationInput,
    DiarizationOutput,
    PreprocessInput,
    PreprocessOutput,
    SpeakerDiarizationStage,
    TranscriptAlignmentStage,
    TranscriptionInput,
    TranscriptionOutput,
    WhisperTranscriptionStage,
)
from voxmind.services.speech.whisper_provider import FasterWhisperProvider
from voxmind.services.storage.interfaces import StorageBackend
from voxmind.workers.task_runner import JobStatus, PipelineStage, TaskRunner, wait_for_completion

# `dispatch()`'s parameter and `get_result()`'s return are typed as the
# generic `BaseModel` in the approved TaskRunner Protocol (workers/
# task_runner.py) - deliberately so every stage shares one contract
# regardless of its own specific input/output types. That means each
# concrete stage class's more specific `run(SpecificInput) -> SpecificOutput`
# signature isn't a strict structural subtype of the Protocol's
# `run(BaseModel) -> BaseModel` under mypy's variance rules, even though it
# is exactly correct at runtime (Python doesn't check this) and is what the
# project's stage-typing requirement asks for. `cast()` below documents that
# gap explicitly at each call site rather than loosening any real type.

logger = structlog.get_logger(__name__)


class AudioService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        storage: StorageBackend,
        task_runner: TaskRunner,
        settings: Settings,
        whisper_provider: FasterWhisperProvider,
        diarization_provider: PyannoteDiarizationProvider,
    ) -> None:
        self._session = session
        self._storage = storage
        self._task_runner = task_runner
        self._settings = settings
        # Injected rather than constructed here: both wrap a lazily-loaded
        # ML model (`self._model`/`self._pipeline`) that must survive across
        # requests - a fresh provider per request would reload the model
        # every single time. Callers get these from cached singleton
        # factories (see api/deps.py).
        self._whisper_provider = whisper_provider
        self._diarization_provider = diarization_provider
        self._conversations = ConversationService(session)
        self._audio_assets = AudioAssetRepository(session)
        self._jobs = AudioProcessingJobRepository(session)
        self._messages = MessageRepository(session)
        self._transcripts = TranscriptRepository(session)

    # --- Ingestion -----------------------------------------------------

    async def upload_audio(
        self,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        filename: str | None,
        raw_bytes: bytes,
    ) -> AudioAsset:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)

        detected_format = validate_audio_upload(
            header=raw_bytes[:64], size_bytes=len(raw_bytes), settings=self._settings
        )
        clean_filename = safe_filename(filename, fallback_extension=detected_format)

        asset = await self._audio_assets.create(
            session_id=conversation_id,
            storage_key="",  # filled in below once we know the generated id
            kind="original",
            original_filename=clean_filename,
            content_type=f"audio/{detected_format}",
            size_bytes=len(raw_bytes),
        )
        storage_key = f"conversations/{conversation_id}/audio/{asset.id}/original.{detected_format}"
        await self._storage.upload(storage_key, raw_bytes, content_type=f"audio/{detected_format}")
        asset.storage_key = storage_key
        await self._session.commit()
        logger.info(
            "audio_ingested",
            conversation_id=str(conversation_id),
            audio_asset_id=str(asset.id),
            format=detected_format,
            size_bytes=len(raw_bytes),
        )
        return asset

    async def list_audio(self, *, conversation_id: uuid.UUID, user_id: uuid.UUID) -> list[AudioAsset]:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        return await self._audio_assets.list_for_conversation(conversation_id)

    async def _get_owned_original_asset(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, audio_asset_id: uuid.UUID
    ) -> AudioAsset:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        asset = await self._audio_assets.get_by_id(audio_asset_id)
        if asset is None or asset.session_id != conversation_id or asset.kind != "original":
            raise NotFoundError("Audio asset not found.")
        return asset

    # --- Pipeline orchestration -----------------------------------------

    async def process_audio(
        self,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        audio_asset_id: uuid.UUID,
        language: str | None = None,
    ) -> AudioProcessingJob:
        original_asset = await self._get_owned_original_asset(
            conversation_id=conversation_id, user_id=user_id, audio_asset_id=audio_asset_id
        )

        job = await self._jobs.create(session_id=conversation_id, audio_asset_id=audio_asset_id)
        await self._session.commit()

        log = logger.bind(
            conversation_id=str(conversation_id),
            audio_asset_id=str(audio_asset_id),
            job_id=str(job.id),
        )
        stage_durations_ms: dict[str, int] = {}

        try:
            processed = await self._run_preprocessing(original_asset, stage_durations_ms, log)
        except AudioProcessingError as exc:
            return await self._fail_job(job, "preprocessing_failed", str(exc), stage_durations_ms, log)

        try:
            transcription = await self._run_transcription(processed, language, stage_durations_ms, log)
        except AudioProcessingError as exc:
            return await self._fail_job(job, "transcription_failed", str(exc), stage_durations_ms, log)

        diarization, diarization_status, diarization_error = await self._run_diarization(
            processed, stage_durations_ms, log
        )

        try:
            turns = await self._run_alignment(transcription, diarization, stage_durations_ms, log)
        except AudioProcessingError as exc:
            return await self._fail_job(job, "alignment_failed", str(exc), stage_durations_ms, log)

        message = await self._persist_results(
            conversation_id=conversation_id,
            audio_asset_id=audio_asset_id,
            transcription=transcription,
            diarization=diarization,
            turns=turns,
        )

        model_versions = {
            "stt": transcription.model_version,
            "diarization": diarization.model_version if diarization_status == "completed" else None,
        }
        await self._jobs.mark_completed(
            job,
            message_id=message.id,
            model_versions=model_versions,
            stage_durations_ms=stage_durations_ms,
            diarization_status=diarization_status,
            diarization_error=diarization_error,
        )
        await self._session.commit()
        log.info("audio_processing_job_completed", diarization_status=diarization_status)
        return job

    async def _fail_job(self, job, error_code, error_message, stage_durations_ms, log) -> AudioProcessingJob:
        await self._jobs.mark_failed(
            job, error_code=error_code, error_message=error_message, stage_durations_ms=stage_durations_ms
        )
        await self._session.commit()
        log.warning("audio_processing_job_failed", error_code=error_code, error=error_message)
        return job

    async def _run_preprocessing(
        self, original_asset: AudioAsset, stage_durations_ms: dict, log
    ) -> PreprocessedAudio:
        processed_key = (
            f"conversations/{original_asset.session_id}/audio/{original_asset.id}/processed.wav"
        )
        stage = AudioPreprocessingStage(self._storage, self._settings)
        started = time.monotonic()
        handle = await self._task_runner.dispatch(
            cast(PipelineStage, stage),
            PreprocessInput(
                original_storage_key=original_asset.storage_key, processed_storage_key=processed_key
            ),
        )
        status_handle = await wait_for_completion(self._task_runner, handle.job_id)
        stage_durations_ms["preprocessing"] = round((time.monotonic() - started) * 1000)
        if status_handle.status == JobStatus.FAILED:
            log.warning("audio_preprocessing_failed", error=status_handle.error)
            raise AudioProcessingError(status_handle.error or "Audio preprocessing failed.")
        result = cast(PreprocessOutput, await self._task_runner.get_result(handle.job_id))
        return result.audio

    async def _run_transcription(
        self, processed: PreprocessedAudio, language: str | None, stage_durations_ms: dict, log
    ) -> TranscriptionResult:
        stage = WhisperTranscriptionStage(self._storage, self._whisper_provider)
        started = time.monotonic()
        handle = await self._task_runner.dispatch(
            cast(PipelineStage, stage),
            TranscriptionInput(
                processed_storage_key=processed.storage_key,
                sample_rate=processed.sample_rate,
                language=language,
            ),
        )
        status_handle = await wait_for_completion(self._task_runner, handle.job_id)
        stage_durations_ms["transcription"] = round((time.monotonic() - started) * 1000)
        if status_handle.status == JobStatus.FAILED:
            log.warning("audio_transcription_failed", error=status_handle.error)
            raise AudioProcessingError(status_handle.error or "Speech-to-text transcription failed.")
        result = cast(TranscriptionOutput, await self._task_runner.get_result(handle.job_id))
        return result.transcription

    async def _run_diarization(
        self, processed: PreprocessedAudio, stage_durations_ms: dict, log
    ) -> tuple[DiarizationResult, str, str | None]:
        if not self._settings.HUGGINGFACE_TOKEN:
            log.info("audio_diarization_unavailable", reason="no_huggingface_token")
            empty = DiarizationResult(speaker_segments=[], num_speakers=0, model_version="none")
            return empty, "unavailable", "HUGGINGFACE_TOKEN is not configured - see docs/audio.md."

        stage = SpeakerDiarizationStage(self._storage, self._diarization_provider)
        started = time.monotonic()
        handle = await self._task_runner.dispatch(
            cast(PipelineStage, stage),
            DiarizationInput(
                processed_storage_key=processed.storage_key, sample_rate=processed.sample_rate
            ),
        )
        status_handle = await wait_for_completion(self._task_runner, handle.job_id)
        stage_durations_ms["diarization"] = round((time.monotonic() - started) * 1000)
        if status_handle.status == JobStatus.FAILED:
            log.warning("audio_diarization_failed", error=status_handle.error)
            empty = DiarizationResult(speaker_segments=[], num_speakers=0, model_version="none")
            return empty, "failed", status_handle.error
        result = cast(DiarizationOutput, await self._task_runner.get_result(handle.job_id))
        return result.diarization, "completed", None

    async def _run_alignment(
        self,
        transcription: TranscriptionResult,
        diarization: DiarizationResult,
        stage_durations_ms: dict,
        log,
    ) -> list[AlignedTurn]:
        stage = TranscriptAlignmentStage()
        started = time.monotonic()
        handle = await self._task_runner.dispatch(
            cast(PipelineStage, stage),
            AlignmentInput(
                transcript_segments=transcription.segments, speaker_segments=diarization.speaker_segments
            ),
        )
        status_handle = await wait_for_completion(self._task_runner, handle.job_id)
        stage_durations_ms["alignment"] = round((time.monotonic() - started) * 1000)
        if status_handle.status == JobStatus.FAILED:
            log.warning("audio_alignment_failed", error=status_handle.error)
            raise AudioProcessingError(status_handle.error or "Transcript/diarization alignment failed.")
        result = cast(AlignmentOutput, await self._task_runner.get_result(handle.job_id))
        return result.turns

    async def _persist_results(
        self,
        *,
        conversation_id: uuid.UUID,
        audio_asset_id: uuid.UUID,
        transcription: TranscriptionResult,
        diarization: DiarizationResult,
        turns: list[AlignedTurn],
    ) -> Message:
        message = await self._messages.create(
            session_id=conversation_id, role="user", content=transcription.text
        )
        message.audio_asset_id = audio_asset_id
        await self._session.flush()

        await self._transcripts.add_transcript_segments(
            message_id=message.id,
            segments=transcription.segments,
            model_version=transcription.model_version,
        )
        if diarization.speaker_segments:
            await self._transcripts.add_speaker_segments(
                message_id=message.id,
                segments=diarization.speaker_segments,
                model_version=diarization.model_version,
            )
        await self._transcripts.add_aligned_turns(message_id=message.id, turns=turns)
        return message

    # --- Status / results retrieval --------------------------------------

    async def get_processing_job(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, job_id: uuid.UUID
    ) -> AudioProcessingJob:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        job = await self._jobs.get_by_id(job_id)
        if job is None or job.session_id != conversation_id:
            raise NotFoundError("Processing job not found.")
        return job

    async def list_processing_jobs(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, audio_asset_id: uuid.UUID
    ) -> list[AudioProcessingJob]:
        await self._get_owned_original_asset(
            conversation_id=conversation_id, user_id=user_id, audio_asset_id=audio_asset_id
        )
        return await self._jobs.list_for_audio_asset(audio_asset_id)

    async def get_transcript(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID
    ):
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        message = await self._messages.get_by_id(message_id)
        if message is None or message.session_id != conversation_id:
            raise NotFoundError("Message not found.")
        segments = await self._transcripts.get_transcript_segments(message_id)
        speakers = await self._transcripts.get_speaker_segments(message_id)
        turns = await self._transcripts.get_aligned_turns(message_id)
        return segments, speakers, turns
