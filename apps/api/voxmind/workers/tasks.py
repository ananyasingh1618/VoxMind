"""The one real Celery task behind every `CeleryTaskRunner` dispatch
(Phase 9). A single generic task - not one task per pipeline stage - looks
up the stage by name in `stage_registry.STAGE_REGISTRY`; this is the
"appropriate task boundary" the pipeline actually needs (one Celery
message per `PipelineStage.run()` call, matching exactly what
`InProcessTaskRunner.dispatch()` already does per call), not one task per
tiny sub-operation and not a second parallel pipeline implementation.

Celery cannot invoke an `async def` callable, so this is a plain `def`
whose body is a single `asyncio.run(...)` call - the boundary ADR 0002 and
`workers/task_runner.py`'s `CeleryTaskRunner` docstring specified before
any of this existed.
"""
from __future__ import annotations

import asyncio
import time
import uuid

import structlog
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from voxmind.core.config import get_settings
from voxmind.repositories.pipeline_run_repository import PipelineRunRepository
from voxmind.workers.celery_app import celery_app
from voxmind.workers.db import worker_session_scope
from voxmind.workers.json_safe import dump_json_safe
from voxmind.workers.stage_registry import STAGE_REGISTRY

logger = structlog.get_logger(__name__)

# Only genuinely transient, infra-level failures are retried - a dropped
# connection talking to storage/a model endpoint. Everything else (missing
# credentials, invalid input, a stage's own domain exceptions like
# AudioProcessingError/UnsupportedAudioFormatError, validation errors) is a
# permanent condition retrying can't fix and must fail immediately - see
# the explicit "do not blindly retry" list in the Phase 9 spec.
#
# `celery.exceptions.SoftTimeLimitExceeded` (raised when
# CELERY_TASK_SOFT_TIME_LIMIT_SECONDS is exceeded - see celery_app.py) is
# always classified PERMANENT, never retried: a stage that structurally
# took too long will very likely take too long again. It has its own
# dedicated `except` clause rather than falling into the generic
# transient/permanent tuple check below, and it is handled in TWO places -
# both real, both verified by testing:
#   1. Inside `_execute_pipeline_stage_async`'s own try/except, for a stage
#      doing tight, non-yielding CPU work - the signal lands inside the
#      coroutine's own frame.
#   2. Around the `asyncio.run(...)` call in `execute_pipeline_stage`
#      itself, for the (realistically far more common, since almost every
#      real stage awaits real I/O) case where the signal instead fires
#      inside asyncio's own event-loop machinery and propagates straight
#      out of `asyncio.run()` - confirmed for real by
#      test_real_soft_time_limit_kills_a_genuinely_slow_stage, which caught
#      a genuine gap: an earlier, single-try version of this task left the
#      `pipeline_runs` row stuck at "running" forever whenever a stage
#      timed out while blocked on I/O.
# The hard limit (CELERY_TASK_TIME_LIMIT_SECONDS, 30s after the soft limit
# by default) remains a real, honest, documented residual gap: it exists
# only as a backstop for code that isn't interruptible by the soft limit's
# signal at all (e.g. stuck in a non-yielding C call with no Python bytecode
# boundary to deliver the signal at). If that backstop ever fires, this
# task's Python code never runs at all - the OS kills the worker child
# process directly - so the `pipeline_runs` row is left at "running". See
# docs/celery.md's "Known limitation: hard time limit" section.
_TRANSIENT_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError, OSError)


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, _TRANSIENT_EXCEPTION_TYPES)


@celery_app.task(bind=True, name="voxmind.execute_pipeline_stage")
def execute_pipeline_stage(self: Task, job_id: str) -> str:
    try:
        return asyncio.run(_execute_pipeline_stage_async(self, job_id))
    except SoftTimeLimitExceeded:
        # A real, verified-by-testing gap in the naive version of this:
        # billiard's soft-time-limit signal handler fires wherever process
        # execution happens to be when the timer expires, which for any
        # stage blocked in real I/O (asyncio.sleep, a socket read, Whisper's
        # asyncio.to_thread() call, ...) is *inside asyncio's own event-loop
        # machinery* (its internal selector.select() call), not inside the
        # awaited coroutine's own frame - so the exception propagates
        # straight out of asyncio.run() instead of ever reaching the
        # `except SoftTimeLimitExceeded` inside
        # _execute_pipeline_stage_async, which only actually fires for a
        # stage doing tight, non-yielding CPU work. Confirmed for real via
        # tests/integration/test_celery_task_runner.py::
        # test_real_soft_time_limit_kills_a_genuinely_slow_stage - the
        # naive single-try version left the `pipeline_runs` row stuck at
        # "running" forever. The original event loop is gone/unusable once
        # this exception has propagated, so a *fresh* asyncio.run() (a
        # fresh loop, a fresh worker session) is the only way left to
        # record the real outcome.
        asyncio.run(_mark_timed_out(job_id))
        logger.warning("pipeline_stage_timed_out_at_event_loop_boundary", job_id=job_id)
        return "failed"


async def _mark_timed_out(job_id_str: str) -> None:
    job_id = uuid.UUID(job_id_str)
    async with worker_session_scope() as session:
        repo = PipelineRunRepository(session)
        run = await repo.get(job_id)
        if run is not None and run.status not in ("completed", "failed"):
            await repo.mark_failed(
                run, error="Stage exceeded its time limit (caught at the event-loop boundary)."
            )
            await session.commit()


async def _execute_pipeline_stage_async(task: Task, job_id_str: str) -> str:
    job_id = uuid.UUID(job_id_str)
    settings = get_settings()

    async with worker_session_scope() as session:
        repo = PipelineRunRepository(session)
        run = await repo.get(job_id)
        if run is None:
            # dispatch() always creates the row before enqueueing (see
            # CeleryTaskRunner) - reaching this means the row was somehow
            # deleted between enqueue and execution, not a normal outcome.
            logger.error("pipeline_run_not_found", job_id=job_id_str)
            return "missing"

        log = logger.bind(
            job_id=job_id_str,
            celery_task_id=task.request.id,
            stage_name=run.stage_name,
            queue=run.queue,
            correlation_id=run.correlation_id,
            retries=task.request.retries,
            worker_hostname=getattr(task.request, "hostname", None),
        )

        # Idempotency guard against task redelivery (task_acks_late=True
        # means a worker killed mid-task leaves the message unacked and it
        # gets redelivered to another worker) - a terminal row is a no-op,
        # never re-executed, so redelivery can never create duplicate
        # downstream persistence.
        if run.status in ("completed", "failed"):
            log.info("pipeline_run_already_terminal_skipping_redelivery", status=run.status)
            return run.status

        # Real poison-pill bound (final hardening pass): `delivery_count`
        # (incremented in mark_running(), see PipelineRunRepository) counts
        # every genuine execution attempt this row has ever had - first
        # dispatch, in-process retries, and broker-redelivery-after-
        # worker-loss alike. CELERY_TASK_MAX_RETRIES already bounds the
        # in-process-retry path on its own (a caught, classified transient
        # exception); this is the backstop for the path that bounds
        # nothing at all today - a worker that crashes/is killed mid-task
        # repeatedly on the exact same input, whose broker-level
        # redelivery never touches retry_count. Checked before mark_running
        # so a run that's already exhausted its budget is never actually
        # re-executed (and never increments delivery_count again for it).
        if run.delivery_count >= settings.CELERY_MAX_DELIVERY_ATTEMPTS:
            await repo.mark_failed(
                run,
                error=(
                    f"poison-pill: exceeded {settings.CELERY_MAX_DELIVERY_ATTEMPTS} delivery "
                    "attempts without ever reaching a terminal state - this input reliably "
                    "crashes or is never completed by whatever worker picks it up; see "
                    "docs/celery.md's poison-pill section."
                ),
            )
            await session.commit()
            log.error("pipeline_run_exceeded_max_delivery_attempts", status="failed", delivery_count=run.delivery_count)
            return "failed"

        registration = STAGE_REGISTRY.get(run.stage_name)
        if registration is None:
            await repo.mark_failed(run, error=f"Unknown stage_name: {run.stage_name!r}")
            await session.commit()
            log.error("pipeline_stage_unknown", status="failed")
            return "failed"

        await repo.mark_running(run, celery_task_id=task.request.id)
        await session.commit()
        # Worker orphan/fencing (final hardening pass): the exact
        # started_at THIS execution attempt itself just set - the fencing
        # token every terminal write below must present unchanged for its
        # write to actually apply. See PipelineRunRepository.
        # mark_completed_if_still_owned()'s docstring for the full design.
        # Captured once, right after mark_running() succeeds, specifically
        # so a long stage.run() call happening in between can't let the
        # token go stale out from under this same attempt.
        expected_started_at = run.started_at
        log.info("pipeline_stage_started")
        started = time.monotonic()

        try:
            stage = await registration.build(settings, session)
            if stage is None:
                applied = await repo.mark_failed_if_still_owned(
                    run,
                    error=f"Stage {run.stage_name!r} is currently unavailable "
                    "(no active trained model, or no provider configured).",
                    expected_started_at=expected_started_at,
                )
                await session.commit()
                duration_ms = round((time.monotonic() - started) * 1000)
                if not applied:
                    log.warning("pipeline_run_write_fenced_out_superseded", status="superseded", duration_ms=duration_ms)
                    return "superseded"
                log.warning("pipeline_stage_unavailable", status="failed", duration_ms=duration_ms)
                return "failed"

            stage_input = registration.input_model.model_validate(run.input_json)
            output = await stage.run(stage_input)
            # Deliberately inside this try, not after it (a real bug found
            # via test_real_worker_runs_tts_synthesis: this used to sit
            # after the try/except entirely, so a serialization failure
            # here - which genuinely happened for real binary audio bytes,
            # see json_safe.py - escaped uncaught and left the row stuck at
            # "running" forever, never reaching mark_failed at all).
            output_json = dump_json_safe(output)
        except SoftTimeLimitExceeded as exc:
            # Explicit, distinct branch (not just "not in the transient
            # tuple") so this classification can't silently change if
            # someone edits _TRANSIENT_EXCEPTION_TYPES later: a task that
            # structurally took too long is a permanent condition, not
            # something a retry is likely to fix. Reaches a real terminal
            # "failed" state here, in-process, before the hard time limit
            # (a backstop, not the primary mechanism) would ever need to
            # kill the worker child outright - see the module-level comment
            # above `_TRANSIENT_EXCEPTION_TYPES` for the known limitation
            # if the hard limit does fire instead.
            duration_ms = round((time.monotonic() - started) * 1000)
            applied = await repo.mark_failed_if_still_owned(
                run, error=f"Stage exceeded its time limit: {exc}", expected_started_at=expected_started_at
            )
            await session.commit()
            if not applied:
                log.warning("pipeline_run_write_fenced_out_superseded", status="superseded", duration_ms=duration_ms)
                return "superseded"
            log.warning("pipeline_stage_timed_out", status="failed", duration_ms=duration_ms)
            return "failed"
        except Exception as exc:  # noqa: BLE001 - every stage failure must be classified and persisted, never crash the worker
            duration_ms = round((time.monotonic() - started) * 1000)
            transient = _is_transient(exc)
            if transient and task.request.retries < settings.CELERY_TASK_MAX_RETRIES:
                delay = settings.CELERY_TASK_RETRY_BACKOFF_SECONDS * (2**task.request.retries)
                await repo.increment_retry_count(run)
                applied = await repo.reset_to_pending_for_retry_if_still_owned(run, expected_started_at=expected_started_at)
                await session.commit()
                if not applied:
                    # This row was superseded (reconciled to a terminal
                    # state, or already re-owned by another execution)
                    # while this attempt was mid-classification - resuming
                    # it into "pending" now would resurrect a decision
                    # something else already made. Don't ask Celery to
                    # retry a row this worker no longer owns.
                    log.warning("pipeline_run_write_fenced_out_superseded", status="superseded", duration_ms=duration_ms)
                    return "superseded"
                log.warning(
                    "pipeline_stage_transient_failure_retrying",
                    error=str(exc),
                    delay_seconds=delay,
                    duration_ms=duration_ms,
                )
                raise task.retry(exc=exc, countdown=delay) from exc
            applied = await repo.mark_failed_if_still_owned(run, error=str(exc), expected_started_at=expected_started_at)
            await session.commit()
            if not applied:
                log.warning("pipeline_run_write_fenced_out_superseded", status="superseded", duration_ms=duration_ms)
                return "superseded"
            log.warning("pipeline_stage_failed", status="failed", error=str(exc), duration_ms=duration_ms, transient=transient)
            return "failed"

        duration_ms = round((time.monotonic() - started) * 1000)
        applied = await repo.mark_completed_if_still_owned(run, output_json=output_json, expected_started_at=expected_started_at)
        await session.commit()
        if not applied:
            # Real worker-orphan protection, not a hypothetical: proves a
            # stale execution's real, successfully-computed result is
            # discarded rather than resurrecting a row something else
            # (reconciliation, or another worker's redelivery) already
            # moved on from - see docs/celery.md's "Worker orphan/fencing"
            # section and the live test proving this exact scenario.
            log.warning("pipeline_run_write_fenced_out_superseded", status="superseded", duration_ms=duration_ms)
            return "superseded"
        log.info("pipeline_stage_completed", status="completed", duration_ms=duration_ms)
        return "completed"


@celery_app.task(name="voxmind.reconcile_stale_pipeline_runs")
def reconcile_stale_pipeline_runs_task() -> int:
    """The periodic Celery Beat task behind automatic reconciliation
    scheduling (final limitations-clearance pass) - see `workers/
    reconciliation.py`'s module docstring for the full "why Celery Beat"
    reasoning and `celery_app.py`'s `beat_schedule` for the interval.

    A thin, real wrapper - all of the actual logic (the stale-threshold
    query, `FOR UPDATE SKIP LOCKED`, marking rows failed) lives in
    `reconcile_stale_pipeline_runs()` itself, unchanged by the existence of
    this scheduled entrypoint; a human running `python -m voxmind.workers.
    reconciliation` by hand calls the exact same function. Opens its own
    worker-local session (the same pattern `execute_pipeline_stage` uses,
    for the same asyncpg/event-loop reasons - see `workers/db.py`).

    Returns the number of runs reconciled this pass (0 is the normal,
    expected outcome on most ticks - not an error) - visible in Celery's
    own task result/logs for observability, distinct from the `pipeline_
    runs` table itself, which stays internal-only (never exposed over
    HTTP, same guard as everywhere else in this codebase)."""
    from voxmind.workers.reconciliation import reconcile_stale_pipeline_runs

    async def _run() -> int:
        async with worker_session_scope() as session:
            result = await reconcile_stale_pipeline_runs(session)
        if result.reconciled_count:
            logger.warning("scheduled_reconciliation_pass_complete", reconciled_count=result.reconciled_count)
        else:
            logger.info("scheduled_reconciliation_pass_complete", reconciled_count=0)
        return result.reconciled_count

    return asyncio.run(_run())
