"""Phase 9: the authoritative Postgres record behind every `CeleryTaskRunner`
dispatch - exactly the `pipeline_runs` table ADR 0002 and
`workers/task_runner.py`'s `CeleryTaskRunner` docstring specified before any
of this was implemented. Celery's own result backend (Redis) is used only
for a small completion marker Celery needs internally; this table is the
real source of truth for `get_status()`/`get_result()`, so status survives
worker restarts and is queryable even if Redis is temporarily unreachable.

`status` reuses the exact same four values as `workers.task_runner.JobStatus`
(pending/running/completed/failed) - deliberately not a fifth "retrying"
value, since a retry is still "running" from the caller's perspective;
`retry_count` carries the extra detail for observability without adding a
state `JobHandle`'s callers would have to newly handle.

`delivery_count` (final hardening pass): a real, database-backed bound on
uncontrolled broker redelivery, distinct from `retry_count`. `retry_count`
only ever increments on a *caught, in-process* transient failure
(`task.retry()`) - it says nothing about a worker that crashes/is killed
mid-task, since that path never reaches the code that increments it at
all. `delivery_count` increments once per genuine execution attempt
(`PipelineRunRepository.mark_running()`, the one place every attempt -
first dispatch, in-process retry, *and* broker-redelivery-after-worker-
loss - passes through), so it bounds the combined total regardless of
which mechanism produced the redelivery. See `workers/tasks.py`'s
`CELERY_MAX_DELIVERY_ATTEMPTS` check for how it's used, and
docs/celery.md's "Poison-pill bound" section for the full design.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base

VALID_STATUSES = ("pending", "running", "completed", "failed")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    __table_args__ = (CheckConstraint(f"status IN {VALID_STATUSES}", name="valid_status"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stage_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    queue: Mapped[str | None] = mapped_column(String(100), nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        doc="The API request's structlog request_id, captured automatically at dispatch() "
        "time (structlog.contextvars, already bound by RequestContextMiddleware since "
        "Phase 5) - lets a worker-process log line be traced back to the HTTP request "
        "that triggered it, without changing TaskRunner.dispatch()'s signature.",
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    input_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    output_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, doc="Celery's own task UUID - distinct from `id`, useful for "
        "correlating with broker/worker-side logs and tooling."
    )
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    delivery_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="How many times a worker has genuinely started executing this run "
        "(incremented in PipelineRunRepository.mark_running(), covering first "
        "dispatch, in-process retries, and broker-redelivery-after-worker-loss "
        "alike) - the real bound against an uncontrolled poison-pill loop. See "
        "the model docstring and workers/tasks.py.",
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        doc="Python-side default (not just server_default) for the same reason as "
        "EvaluationRun.created_at - see models/evaluation_run.py.",
    )
