"""Final verification pass (Part 5): reconciliation for `pipeline_runs` rows
abandoned by a worker that disappeared mid-task.

**The real gap this closes**: `task_acks_late=True` + `broker_transport_
options["visibility_timeout"]` (see `workers/celery_app.py`'s docstring)
already give this project a real, working self-healing path for the
*common* crash case - a worker SIGKILLed/OOM-killed mid-task leaves its
message unacked, and Redis redelivers it to another worker after
`visibility_timeout` seconds, which re-executes the same `job_id` (the
idempotency guard in `workers/tasks.py` makes this safe - a "running" row is
correctly re-executed, never treated as a no-op) and calls `mark_running()`
again, refreshing `started_at`. Nothing was needed here for that path to
work; it already does.

The gap is the case where redelivery *doesn't* happen at all - the broker
message is genuinely gone (e.g. Redis itself was restarted or flushed
between the crash and the next `visibility_timeout` boundary, or the
message was removed by some other operational action) while the Postgres
row is left sitting at `status="running"` forever, since nothing else in
this codebase ever revisits it. `celery_app.py`'s own module docstring and
`docs/celery.md`'s "Known, honest, NOT fixed limitations" section have
flagged this as real, undone future work since Phase 9.1 - this module is
that work.

**Why a plain reconciliation function + CLI entrypoint, not a Celery beat
schedule**: adding `celery beat` would mean a new long-running scheduler
process/state file this project doesn't currently run anywhere (not in
`docker-compose.full.yml`, not in local dev) - a real new piece of
infrastructure, not a reuse of what's already there. A plain, idempotent,
synchronously-invokable function that an operator (or an external cron
entry, a Docker/Kubernetes scheduled job, or a human running it by hand
after a known incident) calls periodically is the smaller, equally
production-sensible choice, and it can be adopted by a real periodic
scheduler later without changing this function's contract at all. See
docs/DECISIONS/0017 for the full reasoning.

**Why the threshold is derived, not guessed**: see
`CELERY_RECONCILIATION_STALE_THRESHOLD_SECONDS`'s definition in
`core/config.py` for the exact derivation. In short: any row whose
`started_at` is older than the threshold has necessarily survived past
both a full redelivery cycle and one full redelivered execution attempt
without being touched again - the only way that happens is if nothing is
ever going to pick it back up. A row still being legitimately retried
(however many times) keeps refreshing `started_at` on each attempt and
therefore never crosses this threshold.

**Why `status="failed"`, not a new fifth status**: `PipelineRun.status`'s
`CHECK` constraint (`models/pipeline_run.py`) only ever allowed
`pending`/`running`/`completed`/`failed` - the same four values
`workers.task_runner.JobStatus` has used since Phase 9. Adding a distinct
"cancelled"/"reconciled" status would mean a schema migration and new
handling in `CeleryTaskRunner.get_status()` and every caller that maps
`JobStatus` today, for no real behavioral benefit - callers already only
ever distinguish terminal-failed from everything else. The `error` message
set here is a clearly-labeled, greppable marker
(`"reconciliation: ..."`) that lets a reconciled run be told apart from an
ordinary task failure in logs/observability without changing the state
machine. This is the same reasoning `docs/celery.md`'s reliability section
already applies to `task_reject_on_worker_lost` - one well-understood
mechanism, not a second overlapping one.

**Safety properties, all covered by
`tests/unit/test_reconciliation.py`**:
  - Never touches a `pending`, `completed`, or already-`failed` row - the
    query filters on `status == "running"` alone.
  - Never touches a `running` row whose `started_at` is within the
    threshold - a genuinely still-executing task is left completely alone.
  - Idempotent and safe to run repeatedly: once reconciled, a row is
    `failed` (terminal) and will never match the `running`-only filter
    again on a subsequent call.
  - Never corrupts a run a redelivery later still completes: if the
    original message somehow *does* still exist and gets redelivered after
    reconciliation already marked the row `failed`, `workers/tasks.py`'s
    own existing idempotency guard treats any terminal row (`failed` is
    exactly as terminal as `completed`) as a no-op - the same protection
    that already exists for ordinary retry exhaustion, reused here for
    free, not reimplemented.
  - Never touches `pipeline_runs` over HTTP - this module has no route,
    is never imported by any `api/` module, and is covered by the same
    `tests/unit/test_pipeline_run_not_exposed.py` structural guard that
    already locks down every other access path to this table.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings, get_settings
from voxmind.models.pipeline_run import PipelineRun
from voxmind.repositories.pipeline_run_repository import PipelineRunRepository

logger = structlog.get_logger(__name__)

RECONCILIATION_ERROR_PREFIX = "reconciliation: "


@dataclass
class ReconciliationResult:
    """What one reconciliation pass actually did - returned so a caller
    (the CLI entrypoint, a test, a future periodic-task wrapper) can log or
    assert on it without re-querying Postgres itself."""

    reconciled_ids: list[uuid.UUID] = field(default_factory=list)

    @property
    def reconciled_count(self) -> int:
        return len(self.reconciled_ids)


async def reconcile_stale_pipeline_runs(
    session: AsyncSession,
    *,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> ReconciliationResult:
    """Finds every `pipeline_runs` row stuck at `status="running"` whose
    `started_at` is older than `CELERY_RECONCILIATION_STALE_THRESHOLD_
    SECONDS`, and marks each one `failed` with a clearly-labeled
    reconciliation error. Returns which rows (if any) were reconciled.

    `now` is only ever overridden by tests (to deterministically simulate
    "time has passed" without a real sleep) - production call sites always
    use the real current time.
    """
    settings = settings or get_settings()
    reference_time = now or datetime.now(timezone.utc)
    cutoff = reference_time - timedelta(seconds=settings.CELERY_RECONCILIATION_STALE_THRESHOLD_SECONDS)

    result = await session.execute(
        select(PipelineRun).where(PipelineRun.status == "running", PipelineRun.started_at < cutoff)
    )
    stale_runs = list(result.scalars())

    repo = PipelineRunRepository(session)
    reconciled_ids: list[uuid.UUID] = []
    for run in stale_runs:
        age_seconds = (reference_time - run.started_at).total_seconds() if run.started_at else None
        await repo.mark_failed(
            run,
            error=(
                f"{RECONCILIATION_ERROR_PREFIX}no update for over "
                f"{settings.CELERY_RECONCILIATION_STALE_THRESHOLD_SECONDS}s while status=running "
                "(presumed abandoned - owning worker likely crashed/restarted without the task "
                "message ever being redelivered); see docs/celery.md's reconciliation section."
            ),
        )
        logger.warning(
            "pipeline_run_reconciled_stale",
            job_id=str(run.id),
            stage_name=run.stage_name,
            age_seconds=age_seconds,
            threshold_seconds=settings.CELERY_RECONCILIATION_STALE_THRESHOLD_SECONDS,
        )
        reconciled_ids.append(run.id)

    await session.commit()
    return ReconciliationResult(reconciled_ids=reconciled_ids)


async def _main() -> None:
    """CLI entrypoint - `python -m voxmind.workers.reconciliation`. Opens
    its own worker-local session (the same `NullPool` engine
    `workers/tasks.py` uses, for the same asyncpg/event-loop reasons - see
    `workers/db.py`), runs one reconciliation pass, and logs the outcome.
    Intended to be invoked periodically by an external scheduler (host
    cron, a Docker/Kubernetes scheduled job, or manually) - see
    docs/celery.md for real invocation examples. Safe to run repeatedly;
    a no-op run (nothing stale found) is not an error."""
    from voxmind.workers.db import worker_session_scope

    async with worker_session_scope() as session:
        result = await reconcile_stale_pipeline_runs(session)

    if result.reconciled_count:
        logger.warning("reconciliation_pass_complete", reconciled_count=result.reconciled_count)
    else:
        logger.info("reconciliation_pass_complete", reconciled_count=0)


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
