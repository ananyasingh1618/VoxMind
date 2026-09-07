from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.pipeline_run import PipelineRun


class PipelineRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        stage_name: str,
        input_json: dict,
        queue: str | None = None,
        correlation_id: str | None = None,
    ) -> PipelineRun:
        row = PipelineRun(
            stage_name=stage_name,
            input_json=input_json,
            queue=queue,
            correlation_id=correlation_id,
            status="pending",
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get(self, job_id: uuid.UUID) -> PipelineRun | None:
        return await self._session.get(PipelineRun, job_id)

    async def mark_running(self, run: PipelineRun, *, celery_task_id: str | None) -> PipelineRun:
        run.status = "running"
        run.celery_task_id = celery_task_id
        run.started_at = datetime.now(timezone.utc)
        await self._session.flush()
        return run

    async def mark_completed(self, run: PipelineRun, *, output_json: dict) -> PipelineRun:
        run.status = "completed"
        run.output_json = output_json
        run.completed_at = datetime.now(timezone.utc)
        await self._session.flush()
        return run

    async def mark_failed(self, run: PipelineRun, *, error: str) -> PipelineRun:
        run.status = "failed"
        run.error = error
        run.completed_at = datetime.now(timezone.utc)
        await self._session.flush()
        return run

    async def increment_retry_count(self, run: PipelineRun) -> PipelineRun:
        run.retry_count += 1
        await self._session.flush()
        return run
