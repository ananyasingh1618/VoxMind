"""Phase 9: real Postgres-backed tests of `CeleryTaskRunner` and the worker
task body.

Three layers, deliberately:
  1. Tests that drive `CeleryTaskRunner.dispatch()/get_status()/get_result()`
     against real Postgres, with the broker call itself monkeypatched -
     these prove the persistence/queue-routing/error-handling contract
     without needing a running broker.
  2. Tests that drive the worker's task body (`_execute_pipeline_stage_async`)
     directly against real Postgres and a real `PipelineStage` - these prove
     status transitions, real output persistence, retry classification, and
     the redelivery/idempotency guard. Most of these reuse the test's own
     `db_session` (via the `worker_reuses_test_session` fixture) rather than
     opening a second real engine, to avoid a known asyncpg/pytest-asyncio
     cross-engine teardown interaction unrelated to the feature under test
     (see conftest.py's note on NullPool).
     `test_worker_uses_its_own_session_not_the_callers` is the one
     exception - proving genuine session separation is its whole point, so
     it exercises the real, separate worker engine.
  3. `test_real_broker_round_trip`, marked `requires_redis`, enqueues
     through a genuine Redis broker and drives a real Celery worker
     subprocess. Skipped (never faked) when no broker is reachable - the
     same honest pattern this project already uses for
     `requires_hf_token`/`requires_llm_credentials`.

Nothing here mocks Celery itself or fabricates a task result.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
import uuid

import pytest
import redis as redis_client
from pydantic import BaseModel
from sqlalchemy import func, select

from voxmind.core.config import get_settings
from voxmind.models.pipeline_run import PipelineRun
from voxmind.repositories.pipeline_run_repository import PipelineRunRepository
from voxmind.services.speech.interfaces import SpeakerSegment, TranscriptSegment
from voxmind.services.speech.stages import AlignmentInput, TranscriptAlignmentStage
from voxmind.workers.task_runner import CeleryTaskRunner, JobStatus


def _redis_is_reachable() -> bool:
    try:
        client = redis_client.Redis.from_url(
            get_settings().REDIS_URL, socket_connect_timeout=1, socket_timeout=1
        )
        client.ping()
        client.close()
        return True
    except Exception:  # noqa: BLE001
        return False


def _alignment_input() -> AlignmentInput:
    return AlignmentInput(
        transcript_segments=[TranscriptSegment(start_ms=0, end_ms=1000, text="hello there", confidence=0.9)],
        speaker_segments=[SpeakerSegment(speaker_label="speaker_0", start_ms=0, end_ms=1200, confidence=None)],
    )


@pytest.fixture
def worker_reuses_test_session(monkeypatch, db_session):
    """Makes `workers.tasks._execute_pipeline_stage_async` open the test's
    own `db_session` instead of a second real engine - see the module
    docstring for why."""
    from voxmind.workers import tasks as tasks_module

    @contextlib.asynccontextmanager
    async def _reuse_test_session():
        yield db_session

    monkeypatch.setattr(tasks_module, "worker_session_scope", _reuse_test_session)


class _NamedStage:
    """`CeleryTaskRunner.dispatch()` only ever reads `stage.name` off the
    object it's given - the real stage is reconstructed inside the worker
    from `stage_registry.STAGE_REGISTRY` (see that module's docstring for
    why). A minimal stand-in carrying just the name is therefore
    sufficient and honest for tests that dispatch a *real*, already-
    registered stage through a *real* worker - it avoids constructing the
    real stage's (possibly heavy) dependencies in the test process, which
    would never actually be used anyway."""

    def __init__(self, name: str) -> None:
        self.name = name


class _StubRetry(Exception):
    """Stands in for `celery.exceptions.Retry`, which Celery's real
    `Task.retry()` raises. Only the retry *decision* is under test here -
    the surrounding Celery machinery is exercised for real by
    `test_real_broker_round_trip` below."""


class _StubTask:
    """Minimal stand-in for the bound Celery `Task` (`self`) - carries the
    same `request.id`/`request.retries` attributes the task body reads."""

    def __init__(self, retries: int = 0) -> None:
        self.request = type("Request", (), {"id": str(uuid.uuid4()), "retries": retries})()
        self.retry_calls = 0

    def retry(self, exc=None, countdown=None):  # noqa: ANN001, ANN201 - mirrors Celery's signature
        self.retry_calls += 1
        raise _StubRetry(str(exc))


@pytest.mark.asyncio
async def test_dispatch_persists_a_pending_run_and_returns_immediately(db_session, monkeypatch):
    """dispatch() must return a handle without waiting for the work, and
    must persist the authoritative row in Postgres *before* enqueueing."""
    sent: list[tuple] = []
    from voxmind.workers import celery_app as celery_module

    monkeypatch.setattr(
        celery_module.celery_app, "send_task", lambda name, args, queue: sent.append((name, args, queue))
    )

    runner = CeleryTaskRunner(db_session, get_settings())
    handle = await runner.dispatch(TranscriptAlignmentStage(), _alignment_input())

    assert handle.status == JobStatus.PENDING
    assert handle.error is None

    run = await PipelineRunRepository(db_session).get(handle.job_id)
    assert run is not None
    assert run.stage_name == "transcript_alignment"
    assert run.status == "pending"
    assert run.output_json is None
    # Only the job id crosses the broker - never the payload, never a secret.
    assert sent == [("voxmind.execute_pipeline_stage", [str(handle.job_id)], run.queue)]
    await db_session.commit()


@pytest.mark.asyncio
async def test_dispatch_captures_the_real_request_correlation_id(db_session, monkeypatch):
    """`RequestContextMiddleware` (Phase 5) has bound a real `request_id` to
    structlog's contextvars for every HTTP request since before Celery
    existed - dispatch() must capture whatever is bound at call time into
    the persisted row, with zero changes to that middleware or to
    `TaskRunner.dispatch()`'s signature. This is the mechanism a real
    production failure gets traced from an HTTP request through to a
    worker-process log line and back."""
    import structlog

    from voxmind.workers import celery_app as celery_module

    monkeypatch.setattr(celery_module.celery_app, "send_task", lambda name, args, queue: None)

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id="req-abc-123")
    try:
        runner = CeleryTaskRunner(db_session, get_settings())
        handle = await runner.dispatch(TranscriptAlignmentStage(), _alignment_input())
    finally:
        structlog.contextvars.clear_contextvars()

    run = await PipelineRunRepository(db_session).get(handle.job_id)
    assert run.correlation_id == "req-abc-123"
    await db_session.commit()


@pytest.mark.asyncio
async def test_dispatch_routes_expensive_stages_to_the_ml_queue(db_session, monkeypatch):
    from voxmind.services.knowledge.stages import ChunkEmbeddingInput, ChunkEmbeddingStage
    from voxmind.workers import celery_app as celery_module

    monkeypatch.setattr(celery_module.celery_app, "send_task", lambda name, args, queue: None)
    settings = get_settings()
    runner = CeleryTaskRunner(db_session, settings)

    ml_handle = await runner.dispatch(ChunkEmbeddingStage(None), ChunkEmbeddingInput(text="hi"))  # type: ignore[arg-type]
    cheap_handle = await runner.dispatch(TranscriptAlignmentStage(), _alignment_input())

    repo = PipelineRunRepository(db_session)
    assert (await repo.get(ml_handle.job_id)).queue == settings.CELERY_QUEUE_ML
    assert (await repo.get(cheap_handle.job_id)).queue == settings.CELERY_QUEUE_DEFAULT
    await db_session.commit()


@pytest.mark.asyncio
async def test_large_input_never_crosses_the_broker_only_a_job_id_does(db_session, monkeypatch):
    """Target 6 (Redis payload safety), as its own explicit, dedicated
    check rather than an incidental assertion inside a differently-named
    test: no matter how large the real input is (here, a stand-in for a
    full transcript/large embedding request - genuine raw audio/binary
    artifacts never pass through a PipelineStage's Pydantic input at all,
    only storage keys referencing them do), the only thing handed to
    `send_task` is the small `pipeline_runs` primary key."""
    from voxmind.services.nlp.stages import NlpAnalysisInput, NlpAnalysisStage

    sent_args: list = []

    from voxmind.workers import celery_app as celery_module

    monkeypatch.setattr(
        celery_module.celery_app,
        "send_task",
        lambda name, args, queue: sent_args.append(args),
    )

    large_text = "x" * 200_000  # 200 KB - stands in for a large real transcript
    runner = CeleryTaskRunner(db_session, get_settings())
    handle = await runner.dispatch(NlpAnalysisStage(None), NlpAnalysisInput(text=large_text))  # type: ignore[arg-type]

    assert len(sent_args) == 1
    (args,) = sent_args
    assert args == [str(handle.job_id)]
    # The only thing that actually crosses the broker is a UUID string -
    # nowhere near the size of the real input, which lives in Postgres only.
    assert sum(len(a) for a in args) < 100
    await db_session.commit()


@pytest.mark.asyncio
async def test_enqueue_failure_marks_the_run_failed_instead_of_leaving_it_orphaned(db_session, monkeypatch):
    """If the broker is unreachable, dispatch() must not leave a row stuck
    in "pending" forever - the caller gets an honest FAILED handle."""
    from voxmind.workers import celery_app as celery_module

    def _boom(name, args, queue):
        raise RuntimeError("broker unreachable")

    monkeypatch.setattr(celery_module.celery_app, "send_task", _boom)

    runner = CeleryTaskRunner(db_session, get_settings())
    handle = await runner.dispatch(TranscriptAlignmentStage(), _alignment_input())

    assert handle.status == JobStatus.FAILED
    assert "broker unreachable" in handle.error
    run = await PipelineRunRepository(db_session).get(handle.job_id)
    assert run.status == "failed"
    assert "Failed to enqueue task" in run.error
    await db_session.commit()


@pytest.mark.asyncio
async def test_get_status_and_get_result_read_authoritative_postgres_state(db_session):
    repo = PipelineRunRepository(db_session)
    run = await repo.create(
        stage_name="transcript_alignment", input_json=_alignment_input().model_dump(mode="json")
    )
    await db_session.commit()
    runner = CeleryTaskRunner(db_session, get_settings())

    assert (await runner.get_status(run.id)).status == JobStatus.PENDING
    with pytest.raises(ValueError):
        await runner.get_result(run.id)

    await repo.mark_completed(
        run,
        output_json={"turns": [{"start_ms": 0, "end_ms": 1000, "text": "hi", "speaker_label": "speaker_0"}]},
    )
    await db_session.commit()

    assert (await runner.get_status(run.id)).status == JobStatus.COMPLETED
    result = await runner.get_result(run.id)
    # A real, typed Pydantic model - not the raw dict - so existing call
    # sites' attribute access keeps working unchanged.
    assert isinstance(result, BaseModel)
    assert result.turns[0].speaker_label == "speaker_0"
    await db_session.commit()


@pytest.mark.asyncio
async def test_unknown_job_id_raises_like_the_in_process_runner(db_session):
    runner = CeleryTaskRunner(db_session, get_settings())
    with pytest.raises(KeyError):
        await runner.get_status(uuid.uuid4())
    with pytest.raises(ValueError):
        await runner.get_result(uuid.uuid4())
    await db_session.commit()


@pytest.mark.asyncio
async def test_worker_executes_the_stage_and_persists_real_output(db_session, worker_reuses_test_session):
    """Drives the worker's task body directly (no broker) against a real
    stage and a real Postgres row."""
    from voxmind.workers.tasks import _execute_pipeline_stage_async

    repo = PipelineRunRepository(db_session)
    run = await repo.create(
        stage_name="transcript_alignment", input_json=_alignment_input().model_dump(mode="json")
    )
    await db_session.commit()

    outcome = await _execute_pipeline_stage_async(_StubTask(), str(run.id))
    assert outcome == "completed"

    await db_session.refresh(run)
    assert run.status == "completed"
    assert run.started_at is not None and run.completed_at is not None
    assert run.output_json["turns"][0]["speaker_label"] == "speaker_0"
    await db_session.commit()


@pytest.mark.asyncio
async def test_worker_records_a_permanent_failure_without_retrying(db_session, worker_reuses_test_session):
    """A deterministic/validation error must fail terminally - the explicit
    "do not blindly retry invalid input" rule."""
    from voxmind.workers.tasks import _execute_pipeline_stage_async

    repo = PipelineRunRepository(db_session)
    run = await repo.create(stage_name="transcript_alignment", input_json={"not": "valid input"})
    await db_session.commit()

    task = _StubTask()
    outcome = await _execute_pipeline_stage_async(task, str(run.id))

    assert outcome == "failed"
    assert task.retry_calls == 0
    await db_session.refresh(run)
    assert run.status == "failed"
    assert run.retry_count == 0
    assert run.error
    await db_session.commit()


@pytest.mark.asyncio
async def test_worker_retries_only_genuinely_transient_failures(
    db_session, worker_reuses_test_session, monkeypatch
):
    from voxmind.workers import tasks as tasks_module

    repo = PipelineRunRepository(db_session)
    run = await repo.create(
        stage_name="transcript_alignment", input_json=_alignment_input().model_dump(mode="json")
    )
    await db_session.commit()

    async def _transient_build(_settings, _session):
        raise ConnectionError("broker/storage blipped")

    original = tasks_module.STAGE_REGISTRY["transcript_alignment"]
    monkeypatch.setitem(
        tasks_module.STAGE_REGISTRY,
        "transcript_alignment",
        type(original)(
            input_model=original.input_model, output_model=original.output_model, build=_transient_build
        ),
    )

    task = _StubTask()
    with pytest.raises(_StubRetry):
        await tasks_module._execute_pipeline_stage_async(task, str(run.id))

    assert task.retry_calls == 1
    await db_session.refresh(run)
    assert run.retry_count == 1
    assert run.status == "pending"  # queued for another attempt, not terminal
    await db_session.commit()


@pytest.mark.asyncio
async def test_transient_failures_stop_retrying_once_max_retries_is_reached(
    db_session, worker_reuses_test_session, monkeypatch
):
    """Bounded retries, not infinite: once `task.request.retries` has
    already reached `CELERY_TASK_MAX_RETRIES`, even a genuinely transient
    error must give up and fail terminally rather than retry again -
    otherwise a persistently-unreachable dependency would retry forever."""
    from voxmind.workers import tasks as tasks_module

    settings = get_settings()
    repo = PipelineRunRepository(db_session)
    run = await repo.create(
        stage_name="transcript_alignment", input_json=_alignment_input().model_dump(mode="json")
    )
    await db_session.commit()

    async def _still_transient_build(_settings, _session):
        raise ConnectionError("still unreachable")

    original = tasks_module.STAGE_REGISTRY["transcript_alignment"]
    monkeypatch.setitem(
        tasks_module.STAGE_REGISTRY,
        "transcript_alignment",
        type(original)(
            input_model=original.input_model, output_model=original.output_model, build=_still_transient_build
        ),
    )

    # Simulates "this is already the Nth retry attempt."
    task = _StubTask(retries=settings.CELERY_TASK_MAX_RETRIES)
    outcome = await tasks_module._execute_pipeline_stage_async(task, str(run.id))

    assert outcome == "failed"
    assert task.retry_calls == 0  # never asked Celery to retry again
    await db_session.refresh(run)
    assert run.status == "failed"
    assert run.status != "pending"
    assert "still unreachable" in run.error
    await db_session.commit()


@pytest.mark.asyncio
async def test_soft_time_limit_exceeded_reaches_a_terminal_failed_state(
    db_session, worker_reuses_test_session, monkeypatch
):
    """Deterministic, no real sleeping and no real Celery time-limit signal
    needed: proves the *classification and persistence* behavior a real
    `SoftTimeLimitExceeded` would trigger (see tasks.py's dedicated except
    clause) - the run reaches a real terminal "failed" state, is never left
    stuck at "running", never retried (a task that structurally took too
    long is a permanent condition), and no corrupt/successful result is
    persisted. `test_real_soft_time_limit_kills_a_genuinely_slow_stage`
    below additionally proves Celery's real signal-based mechanism actually
    raises this exception for real, end to end, through a real worker."""
    from celery.exceptions import SoftTimeLimitExceeded

    from voxmind.workers import tasks as tasks_module

    repo = PipelineRunRepository(db_session)
    run = await repo.create(
        stage_name="transcript_alignment", input_json=_alignment_input().model_dump(mode="json")
    )
    await db_session.commit()

    async def _slow_build(_settings, _session):
        raise SoftTimeLimitExceeded()

    original = tasks_module.STAGE_REGISTRY["transcript_alignment"]
    monkeypatch.setitem(
        tasks_module.STAGE_REGISTRY,
        "transcript_alignment",
        type(original)(
            input_model=original.input_model, output_model=original.output_model, build=_slow_build
        ),
    )

    task = _StubTask()
    outcome = await tasks_module._execute_pipeline_stage_async(task, str(run.id))

    # Terminal, not retried, not stuck "running".
    assert outcome == "failed"
    assert task.retry_calls == 0
    await db_session.refresh(run)
    assert run.status == "failed"
    assert run.status != "running"
    assert run.retry_count == 0
    assert "time limit" in run.error.lower()
    assert run.output_json is None  # no corrupt/successful result persisted
    assert run.completed_at is not None
    # The status abstraction (what an API caller actually sees) agrees.
    runner = CeleryTaskRunner(db_session, get_settings())
    assert (await runner.get_status(run.id)).status == JobStatus.FAILED
    with pytest.raises(ValueError):
        await runner.get_result(run.id)
    await db_session.commit()


@pytest.mark.asyncio
async def test_redelivered_task_is_idempotent_and_never_re_executes(
    db_session, worker_reuses_test_session, monkeypatch
):
    """`task_acks_late=True` means a killed worker's message can be
    redelivered. A terminal row must be a true no-op: not just
    `completed_at` left alone, but genuinely never re-entering the stage
    build/execute path at all - proven here by making the stage's own
    factory raise if it's ever called, so "the guard runs before any
    duplicate work is even attempted" isn't just inferred from unchanged
    timestamps."""
    from voxmind.workers import tasks as tasks_module

    repo = PipelineRunRepository(db_session)
    run = await repo.create(
        stage_name="transcript_alignment", input_json=_alignment_input().model_dump(mode="json")
    )
    real_output = {"turns": [{"start_ms": 0, "end_ms": 1000, "text": "hi", "speaker_label": "speaker_0"}]}
    await repo.mark_completed(run, output_json=real_output)
    await db_session.commit()
    original_completed_at = run.completed_at
    original_celery_task_id = run.celery_task_id
    original_retry_count = run.retry_count

    async def _fail_if_called(_settings, _session):
        raise AssertionError(
            "a terminal pipeline_run must never re-enter stage construction/execution - "
            "the idempotency guard must short-circuit before this point"
        )

    original = tasks_module.STAGE_REGISTRY["transcript_alignment"]
    monkeypatch.setitem(
        tasks_module.STAGE_REGISTRY,
        "transcript_alignment",
        type(original)(
            input_model=original.input_model, output_model=original.output_model, build=_fail_if_called
        ),
    )

    # Simulate two independent redeliveries of the same message.
    outcome_1 = await tasks_module._execute_pipeline_stage_async(_StubTask(), str(run.id))
    outcome_2 = await tasks_module._execute_pipeline_stage_async(_StubTask(), str(run.id))

    assert outcome_1 == "completed"
    assert outcome_2 == "completed"
    await db_session.refresh(run)
    # Nothing about the terminal record changed: no regression, no mutation,
    # no re-run bookkeeping, and - since this is the same primary key on
    # every call - structurally no second row.
    assert run.status == "completed"
    assert run.completed_at == original_completed_at
    assert run.celery_task_id == original_celery_task_id
    assert run.retry_count == original_retry_count
    assert run.output_json == real_output

    row_count = (
        await db_session.execute(
            select(func.count()).select_from(PipelineRun).where(PipelineRun.id == run.id)
        )
    ).scalar_one()
    assert row_count == 1
    await db_session.commit()


@pytest.mark.asyncio
async def test_redelivery_of_an_already_failed_run_never_becomes_success(
    db_session, worker_reuses_test_session, monkeypatch
):
    """State-machine audit: "failed" is exactly as terminal as "completed" -
    a run this codebase's own retry logic already gave up on must never be
    silently resurrected into "completed" by an unrelated broker-level
    redelivery, even if the underlying condition has since resolved and the
    stage would now genuinely succeed. Anything that already observed
    "failed" via get_status()/get_result() must never see a different
    answer later - illegal "failed -> completed" transition, explicitly
    guarded against."""
    from voxmind.workers import tasks as tasks_module

    repo = PipelineRunRepository(db_session)
    run = await repo.create(
        stage_name="transcript_alignment", input_json=_alignment_input().model_dump(mode="json")
    )
    await repo.mark_failed(run, error="permanently gave up earlier")
    await db_session.commit()

    # If this were ever (incorrectly) re-executed, it would genuinely
    # succeed - proving the guard, not the stage, is what prevents it.
    outcome = await tasks_module._execute_pipeline_stage_async(_StubTask(), str(run.id))

    assert outcome == "failed"
    await db_session.refresh(run)
    assert run.status == "failed"
    assert run.output_json is None  # never actually ran, never produced one
    assert run.error == "permanently gave up earlier"  # untouched
    await db_session.commit()


@pytest.mark.asyncio
async def test_missing_run_row_is_reported_not_crashed(db_session, worker_reuses_test_session):
    from voxmind.workers.tasks import _execute_pipeline_stage_async

    outcome = await _execute_pipeline_stage_async(_StubTask(), str(uuid.uuid4()))
    assert outcome == "missing"
    await db_session.commit()


@pytest.mark.asyncio
async def test_redelivery_while_still_running_correctly_re_executes_to_completion(
    db_session, worker_reuses_test_session
):
    """Distinct from the "redelivered after completed" idempotency test
    above: this is the genuine worker-loss scenario Target 2 is actually
    about - a worker started the task, was lost (crashed/OOM/SIGKILL)
    before reaching a terminal state, and the message is redelivered to a
    different worker for the SAME job_id while the row is still "running".

    Unlike a terminal row, "running" is NOT treated as a no-op - the
    idempotency guard only short-circuits on ("completed", "failed"). This
    is the correct, safe behavior: the original attempt never produced a
    result (it never got the chance to), so there is nothing to protect by
    skipping re-execution, and every real `PipelineStage.run()` in this
    codebase is side-effect-idempotent (storage writes go to the same
    deterministic key computed by the caller; no stage writes its own DB
    rows) - re-running from scratch is safe and is what actually recovers
    from the lost worker. The row still ends in one clean terminal state,
    not a corrupted intermediate one."""
    from voxmind.workers.tasks import _execute_pipeline_stage_async

    repo = PipelineRunRepository(db_session)
    run = await repo.create(
        stage_name="transcript_alignment", input_json=_alignment_input().model_dump(mode="json")
    )
    # Simulates "a previous worker attempt got as far as marking this
    # running, then was lost" - not created via the task body, since that
    # attempt (by hypothesis) never got to finish doing anything else.
    await repo.mark_running(run, celery_task_id="a-now-dead-worker-attempt")
    await db_session.commit()

    outcome = await _execute_pipeline_stage_async(_StubTask(), str(run.id))

    assert outcome == "completed"
    await db_session.refresh(run)
    assert run.status == "completed"  # reached a real terminal state, not stuck
    assert run.output_json is not None
    assert run.output_json["turns"][0]["speaker_label"] == "speaker_0"
    # The new attempt's celery_task_id overwrote the dead one's - the record
    # reflects who actually finished the work.
    assert run.celery_task_id != "a-now-dead-worker-attempt"
    await db_session.commit()


@pytest.mark.asyncio
async def test_unknown_stage_name_fails_the_run_with_a_clear_error(db_session, worker_reuses_test_session):
    from voxmind.workers.tasks import _execute_pipeline_stage_async

    repo = PipelineRunRepository(db_session)
    run = await repo.create(stage_name="not_a_real_stage", input_json={})
    await db_session.commit()

    outcome = await _execute_pipeline_stage_async(_StubTask(), str(run.id))

    assert outcome == "failed"
    await db_session.refresh(run)
    assert "Unknown stage_name" in run.error
    await db_session.commit()


@pytest.mark.asyncio
async def test_worker_uses_its_own_session_not_the_callers(db_session):
    """The worker opens a worker-local session (NullPool, its own engine) -
    it must never be handed a request-scoped session. Proven by the fact
    that a row committed by the worker's own session is visible to a
    *different* session afterwards. Deliberately does NOT use
    `worker_reuses_test_session` - proving real session separation is this
    test's whole point."""
    from voxmind.workers.db import worker_session_scope
    from voxmind.workers.tasks import _execute_pipeline_stage_async

    repo = PipelineRunRepository(db_session)
    run = await repo.create(
        stage_name="transcript_alignment", input_json=_alignment_input().model_dump(mode="json")
    )
    await db_session.commit()

    await _execute_pipeline_stage_async(_StubTask(), str(run.id))

    async with worker_session_scope() as independent_session:
        seen = await independent_session.get(PipelineRun, run.id)
        assert seen is not None
        assert seen.status == "completed"


@contextlib.contextmanager
def _real_worker(log_path: str, *, extra_env: dict | None = None, extra_args: list | None = None):
    """Starts a genuine `celery worker` subprocess against the real broker
    and guarantees it's terminated (and its log file closed) afterward,
    however the test exits. `extra_env` is layered onto (not replacing) the
    current environment, so DATABASE_URL/REDIS_URL/etc. are still inherited
    correctly - only used by the timeout test below to flip on the
    test-only slow stage in that one subprocess, never anywhere else.
    `extra_args` appends extra real `celery worker` CLI flags (e.g.
    `--max-tasks-per-child=N`) - used by the memory-robustness recycling
    test below to exercise the exact same recycling flag production sets
    in docker-compose.full.yml, not a different mechanism."""
    log_file = open(log_path, "w")
    env = {**os.environ, **(extra_env or {})}
    worker = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "celery",
            "-A",
            "voxmind.workers.celery_app:celery_app",
            "worker",
            "--loglevel=info",
            "--queues=voxmind.default,voxmind.ml",
            "--concurrency=1",
            "--without-mingle",
            "--without-gossip",
            *(extra_args or []),
        ],
        stdout=log_file,
        stderr=log_file,
        env=env,
    )
    try:
        # A worker takes a moment to boot and subscribe to its queues.
        time.sleep(4)
        yield worker
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
        log_file.close()


async def _poll_until_terminal(db_session, run, *, deadline_seconds: float):
    """Polls a real, separately-committed `PipelineRun` write with a bounded
    deadline - no fixed/flaky sleep, no indefinite loop. `refresh()` is
    required (not a second `.get()`) because SQLAlchemy's identity map would
    otherwise silently return the cached, stale Python object instead of
    re-querying - see the real-bug note this exact issue produced in this
    file's git history."""
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        await db_session.refresh(run)
        if run.status in ("completed", "failed"):
            break
        time.sleep(0.5)
    await db_session.commit()
    return run


@pytest.mark.requires_redis
@pytest.mark.skipif(not _redis_is_reachable(), reason="No reachable Redis broker in this environment.")
@pytest.mark.asyncio
async def test_real_broker_round_trip(db_session):
    """The strongest honest verification available: enqueues through a
    genuine Redis broker to a genuine, subprocess `celery worker`, and
    confirms real completion via Postgres. Skipped, never faked, when no
    broker is reachable."""
    log_path = "/tmp/test_real_broker_round_trip_worker.log"
    with _real_worker(log_path):
        runner = CeleryTaskRunner(db_session, get_settings())
        handle = await runner.dispatch(TranscriptAlignmentStage(), _alignment_input())
        assert handle.status == JobStatus.PENDING

        repo = PipelineRunRepository(db_session)
        run = await repo.get(handle.job_id)
        run = await _poll_until_terminal(db_session, run, deadline_seconds=20)

        assert run.status == "completed", f"{run.error} -- see {log_path}"
        result = await runner.get_result(handle.job_id)
        assert result.turns[0].speaker_label == "speaker_0"


@pytest.mark.requires_redis
@pytest.mark.skipif(not _redis_is_reachable(), reason="No reachable Redis broker in this environment.")
@pytest.mark.asyncio
async def test_real_soft_time_limit_kills_a_genuinely_slow_stage(db_session):
    """The real end-to-end proof behind
    test_soft_time_limit_exceeded_reaches_a_terminal_failed_state's
    deterministic simulation: a genuinely slow stage, running inside a real
    worker subprocess, is really interrupted by Celery's real signal-based
    soft time limit - not a mock, not a simulated exception.

    Uses a per-message `soft_time_limit`/`time_limit` override
    (`apply_async(..., soft_time_limit=..., time_limit=...)`, a real,
    documented Celery feature) so this test needs an extremely short
    timeout without touching the actual production defaults
    (CELERY_TASK_SOFT_TIME_LIMIT_SECONDS=270 / TIME_LIMIT_SECONDS=300)
    anywhere. The slow stage itself only exists in the worker subprocess
    when VOXMIND_TEST_ENABLE_SLOW_STAGE=1 is set for that one process (see
    stage_registry.py) - production and every other test are unaffected.
    """
    log_path = "/tmp/test_real_soft_time_limit_worker.log"
    with _real_worker(log_path, extra_env={"VOXMIND_TEST_ENABLE_SLOW_STAGE": "1"}) as worker:
        assert worker.poll() is None, f"worker exited early - see {log_path}"

        repo = PipelineRunRepository(db_session)
        run = await repo.create(stage_name="test_slow_stage", input_json={"sleep_seconds": 30.0})
        await db_session.commit()

        from voxmind.workers.celery_app import celery_app

        # Real, per-message timeout override - the stage sleeps 30s, the
        # soft limit fires at 1s, the hard limit backstop at 2s.
        celery_app.send_task(
            "voxmind.execute_pipeline_stage",
            args=[str(run.id)],
            queue="voxmind.default",
            soft_time_limit=1,
            time_limit=2,
        )

        run = await _poll_until_terminal(db_session, run, deadline_seconds=15)

        assert run.status == "failed", f"expected a real timeout failure - see {log_path}"
        assert run.status != "running"  # never left stuck
        assert run.output_json is None  # no corrupt/successful result
        assert run.error and "time limit" in run.error.lower()
        assert run.completed_at is not None

        runner = CeleryTaskRunner(db_session, get_settings())
        assert (await runner.get_status(run.id)).status == JobStatus.FAILED


# --- Representative real-worker coverage across stage categories ---------
#
# Already covered live above: transcript_alignment (pure/deterministic) and,
# in the previous Phase 9 session, chunk_embedding (a real ML embedding
# model). The four tests below add one real stage from each remaining
# category this pipeline actually has - not all 13 individually, per the
# explicit Phase 9.1 instruction - all proving the same
# registry -> Celery task -> worker -> DB session -> PipelineStage.run() ->
# persistence -> typed result path, none needing external credentials.

TEST_TEXT_DOCUMENT = (
    b"Merge sort is a stable, divide-and-conquer sorting algorithm.\n\n"
    b"It guarantees O(n log n) performance even in the worst case."
)


@pytest.mark.requires_redis
@pytest.mark.skipif(not _redis_is_reachable(), reason="No reachable Redis broker in this environment.")
@pytest.mark.asyncio
async def test_real_worker_runs_document_ingestion(db_session):
    """Ordinary CPU/service-stage category: real parsing + real
    deterministic chunking (services/knowledge/chunking.py), no ML model."""
    from voxmind.services.knowledge.stages import DocumentIngestionInput
    from voxmind.services.storage.factory import build_storage_backend

    settings = get_settings()
    storage = build_storage_backend(settings)
    storage_key = f"test/celery-hardening/{uuid.uuid4()}.md"
    await storage.upload(storage_key, TEST_TEXT_DOCUMENT, content_type="text/markdown")

    with _real_worker("/tmp/test_real_worker_document_ingestion.log"):
        runner = CeleryTaskRunner(db_session, settings)
        handle = await runner.dispatch(
            _NamedStage("document_ingestion"),
            DocumentIngestionInput(storage_key=storage_key, content_type="text/markdown", filename="doc.md"),
        )
        run = await PipelineRunRepository(db_session).get(handle.job_id)
        run = await _poll_until_terminal(db_session, run, deadline_seconds=20)

    assert run.status == "completed", run.error
    result = await runner.get_result(handle.job_id)
    assert len(result.chunks) >= 1
    assert "Merge sort" in result.chunks[0]


@pytest.mark.requires_redis
@pytest.mark.skipif(not _redis_is_reachable(), reason="No reachable Redis broker in this environment.")
@pytest.mark.asyncio
async def test_real_worker_runs_audio_preprocessing(db_session):
    """Storage/persistence-heavy category: real ffmpeg subprocess call,
    real storage download and upload, using the same genuine
    `hello_world.wav` fixture other real-model tests in this codebase use."""
    from pathlib import Path

    from voxmind.services.speech.stages import PreprocessInput
    from voxmind.services.storage.factory import build_storage_backend

    settings = get_settings()
    storage = build_storage_backend(settings)
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "hello_world.wav"
    original_key = f"test/celery-hardening/{uuid.uuid4()}.wav"
    processed_key = f"test/celery-hardening/{uuid.uuid4()}-processed.wav"
    await storage.upload(original_key, fixture_path.read_bytes(), content_type="audio/wav")

    with _real_worker("/tmp/test_real_worker_audio_preprocessing.log"):
        runner = CeleryTaskRunner(db_session, settings)
        handle = await runner.dispatch(
            _NamedStage("audio_preprocessing"),
            PreprocessInput(original_storage_key=original_key, processed_storage_key=processed_key),
        )
        run = await PipelineRunRepository(db_session).get(handle.job_id)
        run = await _poll_until_terminal(db_session, run, deadline_seconds=30)

    assert run.status == "completed", run.error
    result = await runner.get_result(handle.job_id)
    assert result.audio.sample_rate == 16000
    assert result.audio.duration_seconds > 0
    # The real, real-worker-produced processed WAV genuinely exists in storage.
    assert await storage.exists(processed_key)


@pytest.mark.requires_redis
@pytest.mark.skipif(not _redis_is_reachable(), reason="No reachable Redis broker in this environment.")
@pytest.mark.asyncio
async def test_real_worker_runs_rerank(db_session):
    """Retrieval/processing category: a real cross-encoder reranking model
    (the same one docs/rag.md documents finding and fixing a genuine
    float32-precision bug in - see DECISIONS/0007)."""
    from voxmind.services.retrieval.stages import RerankInput

    settings = get_settings()
    with _real_worker("/tmp/test_real_worker_rerank.log"):
        runner = CeleryTaskRunner(db_session, settings)
        handle = await runner.dispatch(
            _NamedStage("rerank"),
            RerankInput(
                query="What is merge sort's worst-case complexity?",
                passages=[
                    "Merge sort guarantees O(n log n) even in the worst case.",
                    "Bananas are a good source of potassium.",
                ],
            ),
        )
        run = await PipelineRunRepository(db_session).get(handle.job_id)
        run = await _poll_until_terminal(db_session, run, deadline_seconds=60)

    assert run.status == "completed", run.error
    result = await runner.get_result(handle.job_id)
    assert len(result.scores) == 2
    # The genuinely relevant passage should score higher than the irrelevant one.
    assert result.scores[0] > result.scores[1]


@pytest.mark.requires_redis
@pytest.mark.skipif(not _redis_is_reachable(), reason="No reachable Redis broker in this environment.")
@pytest.mark.asyncio
async def test_real_worker_runs_tts_synthesis(db_session):
    """Output/TTS category: the real, credential-free local TTS provider
    (facebook/mms-tts-eng) already used by the Phase 5 voice loop."""
    from voxmind.services.tts.stages import TtsSynthesisInput

    settings = get_settings()
    if settings.TTS_PROVIDER != "local_hf":
        pytest.skip(f"TTS_PROVIDER={settings.TTS_PROVIDER!r} - this test only covers the credential-free local provider.")

    with _real_worker("/tmp/test_real_worker_tts_synthesis.log"):
        runner = CeleryTaskRunner(db_session, settings)
        handle = await runner.dispatch(
            _NamedStage("tts_synthesis"), TtsSynthesisInput(text="Testing.")
        )
        run = await PipelineRunRepository(db_session).get(handle.job_id)
        run = await _poll_until_terminal(db_session, run, deadline_seconds=60)

    assert run.status == "completed", run.error
    result = await runner.get_result(handle.job_id)
    assert result.result.duration_ms > 0
    assert len(result.result.audio_bytes) > 0


@pytest.mark.requires_redis
@pytest.mark.skipif(not _redis_is_reachable(), reason="No reachable Redis broker in this environment.")
@pytest.mark.asyncio
async def test_real_worker_recycles_child_processes_without_breaking_task_correctness(db_session):
    """Part 4 (Celery memory robustness) regression coverage: production's
    real `--max-tasks-per-child=50` flag (docker-compose.full.yml) exists to
    bound any slow memory growth from repeated real model loads inside a
    long-lived worker child. This test proves the mechanism itself is safe
    for this codebase's actual tasks - not just that Celery's recycling
    feature exists in the abstract - by forcing several recycle events
    (`--max-tasks-per-child=2`) within one real worker subprocess and
    confirming every dispatched task still completes correctly across the
    recycle boundary.

    A real, one-off local stress run (dispatching 20 real sequential
    chunk_embedding/rerank tasks with no recycling limit at all, then again
    with `--max-tasks-per-child=5`) found no unbounded RSS growth in this
    codebase's real ML stages - memory plateaus after model warm-up. That
    same run also found that recycling too aggressively (a threshold of 5)
    measurably increased peak memory versus no limit at all, because
    reloading torch/transformers/sentence-transformers from a cold child
    process costs more than any leak it prevents over a short run. This is
    why production's threshold (50) stays as-is here rather than being
    lowered "for robustness" - the real evidence points the other way. This
    test only needs to prove recycling doesn't corrupt task results, at a
    threshold low enough to force it within a fast, deterministic test."""
    from voxmind.services.knowledge.stages import ChunkEmbeddingInput

    settings = get_settings()
    with _real_worker(
        "/tmp/test_real_worker_recycling.log",
        extra_args=["--max-tasks-per-child=2"],
    ):
        runner = CeleryTaskRunner(db_session, settings)
        # 6 tasks against a per-child cap of 2 forces at least two real
        # child-process recycle events within this one worker subprocess.
        for i in range(6):
            handle = await runner.dispatch(
                _NamedStage("chunk_embedding"),
                ChunkEmbeddingInput(text=f"recycling regression sample {i}"),
            )
            run = await PipelineRunRepository(db_session).get(handle.job_id)
            run = await _poll_until_terminal(db_session, run, deadline_seconds=30)
            assert run.status == "completed", f"task {i} failed across a recycle boundary: {run.error}"
            result = await runner.get_result(handle.job_id)
            assert len(result.embedding) > 0
