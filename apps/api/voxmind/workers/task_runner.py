"""TaskRunner: the seam between pipeline stages and however they execute.

Every stage (Phase 2 onward: STT, diarization, emotion, retrieval, LLM, TTS)
implements `PipelineStage.run(input) -> output` using only JSON-serializable
Pydantic input/output — no ORM objects, no open sessions, no assumption that
it's running inside the request's event loop. That single constraint is what
lets the exact same stage implementation run under either TaskRunner below
without modification.

Both implementations share one contract: `dispatch()` returns a `JobHandle`
immediately; it is never the caller's job to assume the work is finished when
`dispatch()` returns. Callers poll `get_status()` / fetch `get_result()`
once status is COMPLETED. This is deliberate even for `InProcessTaskRunner`,
whose dispatch() happens to run synchronously today — because callers are
written against the async handle-based contract from day one, swapping to
`CeleryTaskRunner` later is a config change, not a call-site rewrite.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from typing import Protocol

import structlog
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobHandle(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    error: str | None = None


class PipelineStage(Protocol):
    name: str

    async def run(self, input: BaseModel) -> BaseModel: ...


class TaskRunner(Protocol):
    async def dispatch(self, stage: PipelineStage, input: BaseModel) -> JobHandle: ...

    async def get_status(self, job_id: uuid.UUID) -> JobHandle: ...

    async def get_result(self, job_id: uuid.UUID) -> BaseModel: ...


class InProcessTaskRunner:
    """Runs a stage synchronously in the caller's event loop and immediately
    returns a COMPLETED/FAILED handle. This is the only TaskRunner used in
    Phase 1-4; there is no separate worker process, so "dispatch" and
    "execute" happen at the same moment. Results are held in memory for the
    lifetime of the process — fine for a single-instance dev/demo deployment,
    and irrelevant once CeleryTaskRunner (which persists status in
    `pipeline_runs`) takes over.
    """

    def __init__(self) -> None:
        self._statuses: dict[uuid.UUID, JobHandle] = {}
        self._results: dict[uuid.UUID, BaseModel] = {}

    async def dispatch(self, stage: PipelineStage, input: BaseModel) -> JobHandle:
        job_id = uuid.uuid4()
        self._statuses[job_id] = JobHandle(job_id=job_id, status=JobStatus.RUNNING)
        try:
            result = await stage.run(input)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any stage failure must
            # surface as a FAILED job handle, never an unhandled exception in the runner.
            handle = JobHandle(job_id=job_id, status=JobStatus.FAILED, error=str(exc))
            self._statuses[job_id] = handle
            return handle
        self._results[job_id] = result
        handle = JobHandle(job_id=job_id, status=JobStatus.COMPLETED)
        self._statuses[job_id] = handle
        return handle

    async def get_status(self, job_id: uuid.UUID) -> JobHandle:
        if job_id not in self._statuses:
            raise KeyError(f"Unknown job_id: {job_id}")
        return self._statuses[job_id]

    async def get_result(self, job_id: uuid.UUID) -> BaseModel:
        if job_id not in self._results:
            raise ValueError(f"Job {job_id} has no completed result.")
        return self._results[job_id]


class CeleryTaskRunner:
    """Real Celery-backed implementation (Phase 9) of the exact design this
    class's docstring specified back in Phase 1 (see ADR 0002):

    - `dispatch()` persists a `PipelineRun` row (status=pending) *before*
      enqueueing anything, then enqueues a single generic Celery task
      (`voxmind.workers.tasks.execute_pipeline_stage`) carrying only the new
      row's id - never the stage object, never the real input payload - and
      returns a `JobHandle` immediately. If enqueueing itself fails (broker
      unreachable), the row is marked failed right away rather than left
      stuck in "pending" forever (no orphaned job records).
    - The worker reconstructs a fresh, equivalent stage instance from
      `stage.name` via `workers.stage_registry.STAGE_REGISTRY` - see that
      module for why this is safer and simpler than pickling the actual
      stage object through Redis.
    - `get_status()`/`get_result()` read `pipeline_runs` directly - status
      survives worker restarts and doesn't depend on Celery/Redis being
      reachable at query time. `get_result()` reconstructs the real,
      correctly-typed output Pydantic model (not a plain dict) so existing
      call sites' attribute access keeps working unchanged.

    Requires its own `AsyncSession` - the caller's request-scoped session is
    fine for a normal HTTP-request read/write; this class never opens a
    session itself and never crosses one into a worker process.
    """

    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._session = session
        self._settings = settings

    async def dispatch(self, stage: PipelineStage, input: BaseModel) -> JobHandle:
        from voxmind.repositories.pipeline_run_repository import PipelineRunRepository
        from voxmind.workers.stage_registry import queue_for_stage

        queue = queue_for_stage(stage.name, self._settings)
        correlation_id = structlog.contextvars.get_contextvars().get("request_id")

        repo = PipelineRunRepository(self._session)
        run = await repo.create(
            stage_name=stage.name,
            input_json=input.model_dump(mode="json"),
            queue=queue,
            correlation_id=correlation_id,
        )
        await self._session.commit()

        try:
            from voxmind.workers.celery_app import celery_app

            celery_app.send_task("voxmind.execute_pipeline_stage", args=[str(run.id)], queue=queue)
        except Exception as exc:  # noqa: BLE001 - a broker-unreachable failure must be persisted, not raised past dispatch()
            await repo.mark_failed(run, error=f"Failed to enqueue task: {exc}")
            await self._session.commit()
            return JobHandle(job_id=run.id, status=JobStatus.FAILED, error=str(exc))

        return JobHandle(job_id=run.id, status=JobStatus.PENDING)

    async def get_status(self, job_id: uuid.UUID) -> JobHandle:
        from voxmind.repositories.pipeline_run_repository import PipelineRunRepository

        run = await PipelineRunRepository(self._session).get(job_id)
        if run is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        return JobHandle(job_id=run.id, status=JobStatus(run.status), error=run.error)

    async def get_result(self, job_id: uuid.UUID) -> BaseModel:
        from voxmind.repositories.pipeline_run_repository import PipelineRunRepository
        from voxmind.workers.json_safe import load_json_safe
        from voxmind.workers.stage_registry import STAGE_REGISTRY

        run = await PipelineRunRepository(self._session).get(job_id)
        if run is None or run.status != "completed" or run.output_json is None:
            raise ValueError(f"Job {job_id} has no completed result.")
        registration = STAGE_REGISTRY[run.stage_name]
        # `load_json_safe` reverses `workers.json_safe.dump_json_safe`'s
        # base64 wrapping of any `bytes` field (e.g. TtsResult.audio_bytes)
        # before real Pydantic validation - see that module's docstring.
        return load_json_safe(registration.output_model, run.output_json)


async def wait_for_completion(
    task_runner: TaskRunner,
    job_id: uuid.UUID,
    *,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 0.2,
) -> JobHandle:
    """The real implementation of the "callers poll get_status()" contract
    described in this module's docstring - every call site must use this
    instead of checking `get_status()` exactly once.

    `InProcessTaskRunner.dispatch()` already returns a terminal handle, so
    the first check below always wins and this adds no latency there.
    `CeleryTaskRunner.dispatch()` only enqueues the task and returns
    immediately - a real worker in a separate process then needs real,
    sometimes multi-second, wall-clock time (model loading, real ML
    inference) to reach a terminal state. A single post-dispatch check
    races that and loses almost every time, raising "Job has no completed
    result" from `get_result()` even though the task goes on to succeed
    moments later - a real bug found during final end-to-end validation:
    every service's call site had already been written against the
    async handle-based contract, but none of them actually polled.

    `timeout_seconds` defaults to `CELERY_TASK_TIME_LIMIT_SECONDS`'s own
    default (300s, see core/config.py) - a real Celery task that hasn't
    reached a terminal DB state by then has already been killed by its own
    hard time limit, so waiting any longer here would never pay off.
    """
    deadline = time.monotonic() + timeout_seconds
    handle = await task_runner.get_status(job_id)
    while handle.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Job {job_id} did not reach a terminal state within {timeout_seconds}s.")
        await asyncio.sleep(poll_interval_seconds)
        handle = await task_runner.get_status(job_id)
    return handle


def build_task_runner(
    mode: str, *, session: AsyncSession | None = None, settings: Settings | None = None
) -> TaskRunner:
    if mode == "in_process":
        return InProcessTaskRunner()
    if mode == "celery":
        if session is None or settings is None:
            raise ValueError("CeleryTaskRunner requires a session and settings.")
        return CeleryTaskRunner(session, settings)
    raise ValueError(f"Unknown TASK_RUNNER mode: {mode!r}")
