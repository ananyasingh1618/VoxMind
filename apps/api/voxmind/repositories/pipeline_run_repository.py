from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import cast

from sqlalchemy import CursorResult, update
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
        # Real poison-pill bound (final hardening pass) - this is the one
        # place, in production, every genuine execution attempt passes
        # through (first dispatch, an in-process retry, and a broker
        # redelivery after worker loss all reach here identically), so
        # it's the correct single point to count them all. See
        # models/pipeline_run.py's docstring and workers/tasks.py.
        run.delivery_count += 1
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

    async def mark_completed_if_still_owned(
        self, run: PipelineRun, *, output_json: dict, expected_started_at: datetime | None
    ) -> bool:
        """Worker orphan/fencing (final hardening pass): a real, database-
        level compare-and-swap using the row's existing `status` +
        `started_at` pair as the fencing token - no new column, no new
        locking system. `expected_started_at` must be exactly the
        `started_at` value THIS execution attempt itself observed right
        after its own `mark_running()` call succeeded (see workers/
        tasks.py). The `UPDATE` only actually applies if the row is
        *still* `status="running"` with *that exact* `started_at` -
        i.e. nothing else (reconciliation marking it terminal, or another
        worker's redelivery-triggered re-execution refreshing
        `started_at` again) has touched this row since. Returns whether
        the write actually applied; the caller's in-memory `run` object is
        only mutated to match on success, so a rejected write can never
        leave the caller believing its own result was persisted.

        See docs/celery.md's "Worker orphan/fencing" section for the full
        design and why plain Postgres row-locking (no `SELECT ... FOR
        UPDATE` needed on this side) already makes this safe against a
        concurrent reconciliation pass - and
        tests/integration/test_celery_task_runner.py's fencing tests for
        the real, live proof."""
        result = await self._session.execute(
            update(PipelineRun)
            .where(
                PipelineRun.id == run.id,
                PipelineRun.status == "running",
                PipelineRun.started_at == expected_started_at,
            )
            .values(status="completed", output_json=output_json, completed_at=datetime.now(timezone.utc))
        )
        # `AsyncSession.execute()`'s static return type is the generic
        # `Result[Any]` (not overloaded per statement kind), but an UPDATE
        # statement genuinely returns a `CursorResult` at runtime, which
        # is what actually carries `.rowcount` - a real, honest cast
        # documenting the actual runtime type, not a blanket ignore.
        applied = cast(CursorResult, result).rowcount > 0
        if applied:
            run.status = "completed"
            run.output_json = output_json
            run.completed_at = datetime.now(timezone.utc)
        await self._session.flush()
        return applied

    async def mark_failed_if_still_owned(
        self, run: PipelineRun, *, error: str, expected_started_at: datetime | None
    ) -> bool:
        """The `mark_failed` counterpart to `mark_completed_if_still_owned` -
        see its docstring for the full fencing design."""
        result = await self._session.execute(
            update(PipelineRun)
            .where(
                PipelineRun.id == run.id,
                PipelineRun.status == "running",
                PipelineRun.started_at == expected_started_at,
            )
            .values(status="failed", error=error, completed_at=datetime.now(timezone.utc))
        )
        applied = cast(CursorResult, result).rowcount > 0
        if applied:
            run.status = "failed"
            run.error = error
            run.completed_at = datetime.now(timezone.utc)
        await self._session.flush()
        return applied

    async def reset_to_pending_for_retry_if_still_owned(
        self, run: PipelineRun, *, expected_started_at: datetime | None
    ) -> bool:
        """A third fencing point, alongside the two above: a caught,
        genuinely transient failure schedules an in-process Celery retry
        by resetting the row back to `status="pending"` - without fencing,
        a worker whose row was reconciled to a terminal state *while it
        was mid-retry-classification* could resurrect it straight back
        into `pending`, undoing reconciliation's decision just as surely
        as a stale completion could. Same compare-and-swap semantics as
        `mark_completed_if_still_owned`/`mark_failed_if_still_owned`."""
        result = await self._session.execute(
            update(PipelineRun)
            .where(
                PipelineRun.id == run.id,
                PipelineRun.status == "running",
                PipelineRun.started_at == expected_started_at,
            )
            .values(status="pending")
        )
        applied = cast(CursorResult, result).rowcount > 0
        if applied:
            run.status = "pending"
        await self._session.flush()
        return applied
