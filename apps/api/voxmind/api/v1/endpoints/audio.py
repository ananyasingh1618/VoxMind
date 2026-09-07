from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import (
    get_current_user,
    get_diarization_provider,
    get_storage_backend,
    get_task_runner,
    get_whisper_provider,
)
from voxmind.core.config import Settings, get_settings
from voxmind.core.exceptions import AudioTooLargeError
from voxmind.core.rate_limit import rate_limit_by_user
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.audio import (
    AudioAssetOut,
    AudioProcessingJobOut,
    ProcessAudioRequest,
    TranscriptOut,
)
from voxmind.services.audio_service import AudioService
from voxmind.services.speech.diarization_provider import PyannoteDiarizationProvider
from voxmind.services.speech.whisper_provider import FasterWhisperProvider
from voxmind.services.storage.interfaces import StorageBackend
from voxmind.workers.task_runner import TaskRunner

router = APIRouter(prefix="/conversations/{conversation_id}", tags=["audio"])


def get_audio_service(
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(get_storage_backend),
    task_runner: TaskRunner = Depends(get_task_runner),
    whisper_provider: FasterWhisperProvider = Depends(get_whisper_provider),
    diarization_provider: PyannoteDiarizationProvider = Depends(get_diarization_provider),
) -> AudioService:
    return AudioService(
        session,
        storage=storage,
        task_runner=task_runner,
        settings=settings,
        whisper_provider=whisper_provider,
        diarization_provider=diarization_provider,
    )


async def _read_upload_bounded(file: UploadFile, *, max_bytes: int) -> bytes:
    """Reads the upload in chunks, aborting as soon as the configured limit
    is exceeded rather than buffering an arbitrarily large payload into
    memory first and rejecting it only afterward."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise AudioTooLargeError(f"Audio upload exceeds the {max_bytes} byte limit.")
        chunks.append(chunk)
    return b"".join(chunks)


_expensive_rate_limit = Depends(rate_limit_by_user("expensive", "RATE_LIMIT_EXPENSIVE_PER_WINDOW"))


@router.post(
    "/audio",
    response_model=AudioAssetOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[_expensive_rate_limit],
)
async def upload_audio(
    conversation_id: uuid.UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    service: AudioService = Depends(get_audio_service),
) -> AudioAssetOut:
    raw_bytes = await _read_upload_bounded(file, max_bytes=settings.MAX_AUDIO_UPLOAD_BYTES)
    asset = await service.upload_audio(
        conversation_id=conversation_id,
        user_id=current_user.id,
        filename=file.filename,
        raw_bytes=raw_bytes,
    )
    return AudioAssetOut.model_validate(asset)


@router.get("/audio", response_model=list[AudioAssetOut])
async def list_audio(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: AudioService = Depends(get_audio_service),
) -> list[AudioAssetOut]:
    assets = await service.list_audio(conversation_id=conversation_id, user_id=current_user.id)
    return [AudioAssetOut.model_validate(a) for a in assets]


@router.post(
    "/audio/{audio_asset_id}/process",
    response_model=AudioProcessingJobOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[_expensive_rate_limit],
)
async def process_audio(
    conversation_id: uuid.UUID,
    audio_asset_id: uuid.UUID,
    body: ProcessAudioRequest,
    current_user: User = Depends(get_current_user),
    service: AudioService = Depends(get_audio_service),
) -> AudioProcessingJobOut:
    """Runs the full speech pipeline. Under Phase 1-4's InProcessTaskRunner
    this genuinely executes synchronously before responding - the HTTP
    status code is always 201 (the processing *attempt* completed as an
    operation) regardless of whether the job's own `status` field ends up
    "completed" or "failed"; clients branch on that field, exactly as they
    would against a truly asynchronous job API. See docs/audio.md.
    """
    job = await service.process_audio(
        conversation_id=conversation_id,
        user_id=current_user.id,
        audio_asset_id=audio_asset_id,
        language=body.language,
    )
    return AudioProcessingJobOut.model_validate(job)


@router.get("/audio/{audio_asset_id}/processing", response_model=list[AudioProcessingJobOut])
async def list_processing_jobs(
    conversation_id: uuid.UUID,
    audio_asset_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: AudioService = Depends(get_audio_service),
) -> list[AudioProcessingJobOut]:
    jobs = await service.list_processing_jobs(
        conversation_id=conversation_id, user_id=current_user.id, audio_asset_id=audio_asset_id
    )
    return [AudioProcessingJobOut.model_validate(j) for j in jobs]


@router.get("/processing-jobs/{job_id}", response_model=AudioProcessingJobOut)
async def get_processing_job(
    conversation_id: uuid.UUID,
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: AudioService = Depends(get_audio_service),
) -> AudioProcessingJobOut:
    job = await service.get_processing_job(
        conversation_id=conversation_id, user_id=current_user.id, job_id=job_id
    )
    return AudioProcessingJobOut.model_validate(job)


@router.get("/messages/{message_id}/transcript", response_model=TranscriptOut)
async def get_transcript(
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: AudioService = Depends(get_audio_service),
) -> TranscriptOut:
    segments, speakers, turns = await service.get_transcript(
        conversation_id=conversation_id, user_id=current_user.id, message_id=message_id
    )
    return TranscriptOut(
        message_id=message_id,
        transcript_segments=list(segments),
        speaker_segments=list(speakers),
        aligned_turns=list(turns),
    )
