"""Final verification pass (Part 5): real, Postgres-backed tests for
`workers/reconciliation.py`. No broker/worker subprocess is needed for most
of these - reconciliation operates purely on `pipeline_runs` rows, which is
exactly why it's implemented and tested independently of Celery itself.
The one exception (`test_a_reconciled_run_is_never_resurrected_by_a_later_
redelivery`) drives the real worker task body directly, reusing the same
`worker_reuses_test_session` pattern `test_celery_task_runner.py` already
established, to prove end-to-end (not just at the repository layer) that a
reconciled run can never come back to life.

Covers, at minimum, every scenario the final verification pass specified:
normal completed run untouched; active run untouched before threshold;
stale run reconciled; repeated reconciliation idempotent; failed run
untouched; a run with no separate "cancelled" state (this schema has none -
documented, not invented) left untouched; a worker-restart/late-redelivery
scenario doesn't corrupt state; a long-running legitimate task (whose
started_at keeps being refreshed by real redelivery) is never prematurely
reconciled.

**A note on why these assert directly on the in-memory `run`/`stage`
object rather than calling `db_session.refresh(...)` afterward**: within
one SQLAlchemy session, the identity map guarantees that fetching the same
primary key twice returns the *same* Python object with its attributes
already updated in place - `reconcile_stale_pipeline_runs()` runs its own
query against this same `db_session`, so the test's own `run` reference is
already live, no re-fetch needed. This isn't just a style preference: a
`db_session.refresh()` call issued *after* a function that already
committed on the same session was found, while writing this file, to
reproducibly trigger a real (separate, teardown-only, assertion-irrelevant)
asyncpg/NullPool "attached to a different loop" error at test teardown -
the same general class of async engine/event-loop interaction
`tests/conftest.py`'s own module docstring already documents. Relying on
the identity map instead of a redundant refresh sidesteps it entirely
without touching any shared pytest-asyncio configuration.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone

import pytest

from voxmind.core.config import get_settings
from voxmind.repositories.pipeline_run_repository import PipelineRunRepository
from voxmind.workers.reconciliation import RECONCILIATION_ERROR_PREFIX, reconcile_stale_pipeline_runs

THRESHOLD_SECONDS = get_settings().CELERY_RECONCILIATION_STALE_THRESHOLD_SECONDS


async def _make_run(db_session, *, status: str, started_at=None, error: str | None = None):
    repo = PipelineRunRepository(db_session)
    run = await repo.create(stage_name="transcript_alignment", input_json={"turns": []})
    if status == "running":
        await repo.mark_running(run, celery_task_id=None)
    elif status == "completed":
        await repo.mark_running(run, celery_task_id=None)
        await repo.mark_completed(run, output_json={"ok": True})
    elif status == "failed":
        # mark_failed works from any non-terminal state; running first is
        # the realistic path (a task that started, then genuinely failed).
        await repo.mark_running(run, celery_task_id=None)
        await repo.mark_failed(run, error=error or "a genuine, ordinary task failure")
    elif status == "pending":
        pass  # repo.create() already leaves it pending
    else:
        raise ValueError(status)

    if started_at is not None:
        run.started_at = started_at
        await db_session.flush()
    await db_session.commit()
    return run


def _stale_timestamp() -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=THRESHOLD_SECONDS + 60)


def _fresh_timestamp() -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=30)


async def test_completed_run_is_never_touched(db_session):
    run = await _make_run(db_session, status="completed", started_at=_stale_timestamp())

    result = await reconcile_stale_pipeline_runs(db_session)

    assert run.id not in result.reconciled_ids
    assert run.status == "completed"


async def test_failed_run_is_never_touched(db_session):
    run = await _make_run(db_session, status="failed", started_at=_stale_timestamp(), error="a real transient-retry exhaustion")

    result = await reconcile_stale_pipeline_runs(db_session)

    assert run.id not in result.reconciled_ids
    assert run.status == "failed"
    assert run.error == "a real transient-retry exhaustion"  # untouched, not overwritten


async def test_a_run_failed_for_a_cancellation_like_reason_is_never_touched(db_session):
    """This schema (models/pipeline_run.py's CHECK constraint) has exactly
    four statuses - pending/running/completed/failed - with no separate
    "cancelled" value; a deliberately cancelled run is represented as
    status="failed" with a distinguishing error message, same as any other
    permanent failure. Reconciliation must leave it alone exactly like any
    other already-terminal row - documented here rather than assuming a
    status this codebase doesn't actually have."""
    run = await _make_run(db_session, status="failed", started_at=_stale_timestamp(), error="cancelled by user request")

    result = await reconcile_stale_pipeline_runs(db_session)

    assert run.id not in result.reconciled_ids
    assert run.error == "cancelled by user request"


async def test_pending_run_is_never_touched(db_session):
    run = await _make_run(db_session, status="pending")

    result = await reconcile_stale_pipeline_runs(db_session)

    assert run.id not in result.reconciled_ids
    assert run.status == "pending"


async def test_active_running_run_within_the_threshold_is_untouched(db_session):
    run = await _make_run(db_session, status="running", started_at=_fresh_timestamp())

    result = await reconcile_stale_pipeline_runs(db_session)

    assert run.id not in result.reconciled_ids
    assert run.status == "running"


async def test_stale_running_run_is_reconciled(db_session):
    run = await _make_run(db_session, status="running", started_at=_stale_timestamp())

    result = await reconcile_stale_pipeline_runs(db_session)

    assert run.id in result.reconciled_ids
    assert run.status == "failed"
    assert run.error is not None and run.error.startswith(RECONCILIATION_ERROR_PREFIX)
    assert run.completed_at is not None


async def test_reconciliation_is_idempotent_across_repeated_calls(db_session):
    run = await _make_run(db_session, status="running", started_at=_stale_timestamp())

    first = await reconcile_stale_pipeline_runs(db_session)
    assert run.id in first.reconciled_ids
    first_error = run.error

    second = await reconcile_stale_pipeline_runs(db_session)
    third = await reconcile_stale_pipeline_runs(db_session)

    assert run.id not in second.reconciled_ids  # already terminal - never matched again
    assert run.id not in third.reconciled_ids
    assert run.status == "failed"
    assert run.error == first_error  # not re-stamped/overwritten by the later, no-op passes


async def test_a_long_running_legitimate_task_refreshed_by_redelivery_is_not_prematurely_reconciled(db_session):
    """The core correctness property of a started_at-based threshold: a row
    originally started long ago (its `created_at` is old) but whose most
    recent execution attempt (`started_at`, refreshed by mark_running() on
    every real redelivery - see workers/tasks.py) is recent must never be
    treated as abandoned merely because it has existed, in some form, for a
    long time. Only genuinely untouched-since-the-threshold rows qualify."""
    repo = PipelineRunRepository(db_session)
    run = await repo.create(stage_name="transcript_alignment", input_json={"turns": []})
    await repo.mark_running(run, celery_task_id="first-attempt")
    run.started_at = _stale_timestamp()  # first attempt looks old/abandoned in isolation
    await db_session.flush()
    await db_session.commit()

    # A real redelivery re-executes the same row and calls mark_running()
    # again, refreshing started_at - simulated directly here since this
    # test is about the reconciliation query, not the worker task body
    # (which is covered separately, see test_celery_task_runner.py's
    # test_redelivery_while_still_running_correctly_re_executes_to_completion).
    await repo.mark_running(run, celery_task_id="second-attempt-after-redelivery")
    await db_session.commit()
    assert run.started_at is not None
    assert (datetime.now(timezone.utc) - run.started_at).total_seconds() < 5

    result = await reconcile_stale_pipeline_runs(db_session)

    assert run.id not in result.reconciled_ids
    assert run.status == "running"


async def test_only_the_genuinely_stale_rows_are_reconciled_among_a_mix(db_session):
    stale = await _make_run(db_session, status="running", started_at=_stale_timestamp())
    active = await _make_run(db_session, status="running", started_at=_fresh_timestamp())
    completed = await _make_run(db_session, status="completed", started_at=_stale_timestamp())
    pending = await _make_run(db_session, status="pending")
    failed = await _make_run(db_session, status="failed", started_at=_stale_timestamp())

    result = await reconcile_stale_pipeline_runs(db_session)

    assert result.reconciled_ids == [stale.id]
    assert active.status == "running"
    assert completed.status == "completed"
    assert pending.status == "pending"
    assert failed.status == "failed"


@pytest.fixture
def worker_reuses_test_session(monkeypatch, db_session):
    """Same pattern as test_celery_task_runner.py's fixture of the same
    name - makes the real worker task body use this test's own db_session
    instead of opening a second real engine."""
    from voxmind.workers import tasks as tasks_module

    @contextlib.asynccontextmanager
    async def _reuse_test_session():
        yield db_session

    monkeypatch.setattr(tasks_module, "worker_session_scope", _reuse_test_session)


async def test_a_reconciled_run_is_never_resurrected_by_a_later_redelivery(db_session, worker_reuses_test_session):
    """The real, end-to-end "worker restart doesn't corrupt state" proof:
    a run reconciliation has already marked terminal must stay terminal
    even if the original broker message somehow still exists and is
    redelivered afterward - driven through the actual worker task body
    (`_execute_pipeline_stage_async`), not just asserted at the repository
    layer. This reuses `workers/tasks.py`'s own pre-existing idempotency
    guard (a "failed" row is as terminal as "completed") - reconciliation
    doesn't need to invent new protection here, it inherits this one."""
    from voxmind.workers import tasks as tasks_module

    class _StubTask:
        request = type("Request", (), {"id": "late-redelivery-task-id", "retries": 0, "hostname": "test"})()

    run = await _make_run(db_session, status="running", started_at=_stale_timestamp())

    reconciliation_result = await reconcile_stale_pipeline_runs(db_session)
    assert run.id in reconciliation_result.reconciled_ids

    # Simulate the original (long-dead) worker's message finally being
    # redelivered and executed, well after reconciliation already gave up
    # on this row.
    outcome = await tasks_module._execute_pipeline_stage_async(_StubTask(), str(run.id))

    assert outcome == "failed"  # never resurrected into "completed"
    assert run.status == "failed"
    assert run.error is not None and run.error.startswith(RECONCILIATION_ERROR_PREFIX)  # untouched by the late redelivery
    assert run.output_json is None
