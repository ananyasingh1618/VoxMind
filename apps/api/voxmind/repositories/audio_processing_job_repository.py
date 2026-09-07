from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.audio_processing_job import AudioProcessingJob


class AudioProcessingJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, *, session_id: uuid.UUID, audio_asset_id: uuid.UUID) -> AudioProcessingJob:
        job = AudioProcessingJob(session_id=session_id, audio_asset_id=audio_asset_id, status="running")
        self._session.add(job)
        await self._session.flush()
        return job

    async def mark_completed(
        self,
        job: AudioProcessingJob,
        *,
        message_id: uuid.UUID,
        model_versions: dict,
        stage_durations_ms: dict,
        diarization_status: str,
        diarization_error: str | None,
    ) -> AudioProcessingJob:
        job.status = "completed"
        job.message_id = message_id
        job.model_versions = model_versions
        job.stage_durations_ms = stage_durations_ms
        job.diarization_status = diarization_status
        job.diarization_error = diarization_error
        job.completed_at = datetime.now(timezone.utc)
        await self._session.flush()
        return job

    async def mark_failed(
        self, job: AudioProcessingJob, *, error_code: str, error_message: str, stage_durations_ms: dict
    ) -> AudioProcessingJob:
        job.status = "failed"
        job.error_code = error_code
        job.error_message = error_message
        job.stage_durations_ms = stage_durations_ms
        job.completed_at = datetime.now(timezone.utc)
        await self._session.flush()
        return job

    async def get_by_id(self, job_id: uuid.UUID) -> AudioProcessingJob | None:
        return await self._session.get(AudioProcessingJob, job_id)

    async def list_for_audio_asset(self, audio_asset_id: uuid.UUID) -> list[AudioProcessingJob]:
        stmt = (
            select(AudioProcessingJob)
            .where(AudioProcessingJob.audio_asset_id == audio_asset_id)
            .order_by(AudioProcessingJob.created_at.desc())
        )
        return list((await self._session.execute(stmt)).scalars().all())
