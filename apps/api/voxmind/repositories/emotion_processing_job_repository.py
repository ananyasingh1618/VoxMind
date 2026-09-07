from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.emotion_processing_job import EmotionProcessingJob


class EmotionProcessingJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, *, session_id: uuid.UUID, message_id: uuid.UUID) -> EmotionProcessingJob:
        job = EmotionProcessingJob(session_id=session_id, message_id=message_id, status="running")
        self._session.add(job)
        await self._session.flush()
        return job

    async def mark_unavailable(self, job: EmotionProcessingJob, *, reason: str) -> EmotionProcessingJob:
        job.status = "unavailable"
        job.error_message = reason
        job.completed_at = datetime.now(timezone.utc)
        await self._session.flush()
        return job

    async def mark_completed(
        self,
        job: EmotionProcessingJob,
        *,
        model_version_id: uuid.UUID,
        turns_processed: int,
        stage_durations_ms: dict,
    ) -> EmotionProcessingJob:
        job.status = "completed"
        job.model_version_id = model_version_id
        job.turns_processed = turns_processed
        job.stage_durations_ms = stage_durations_ms
        job.completed_at = datetime.now(timezone.utc)
        await self._session.flush()
        return job

    async def mark_failed(
        self, job: EmotionProcessingJob, *, error_code: str, error_message: str, stage_durations_ms: dict
    ) -> EmotionProcessingJob:
        job.status = "failed"
        job.error_code = error_code
        job.error_message = error_message
        job.stage_durations_ms = stage_durations_ms
        job.completed_at = datetime.now(timezone.utc)
        await self._session.flush()
        return job

    async def get_by_id(self, job_id: uuid.UUID) -> EmotionProcessingJob | None:
        return await self._session.get(EmotionProcessingJob, job_id)

    async def list_for_message(self, message_id: uuid.UUID) -> list[EmotionProcessingJob]:
        stmt = (
            select(EmotionProcessingJob)
            .where(EmotionProcessingJob.message_id == message_id)
            .order_by(EmotionProcessingJob.created_at.desc())
        )
        return list((await self._session.execute(stmt)).scalars().all())
