# ADR 0002: TaskRunner boundary

## Decision

Every future pipeline stage (STT, diarization, emotion, retrieval, LLM generation, TTS) implements a single `run(input) -> output` method using only JSON-serializable Pydantic models - no ORM objects, no open DB sessions, no assumption about which event loop or process it runs in. A `TaskRunner` dispatches stages and returns a `JobHandle` immediately; callers poll `get_status()`/`get_result()` rather than assuming synchronous completion, even when the current implementation happens to complete synchronously.

## Why not just call stage functions directly and add Celery later?

Because "add Celery later" without this constraint means rewriting every call site: a direct function call assumes synchronous completion, but a Celery task must be dispatched and polled/awaited asynchronously, and Celery cannot invoke an `async def` callable at all - it needs a plain `def` task whose body runs its own event loop (`asyncio.run(...)`). Establishing the async, handle-based contract now - even though `InProcessTaskRunner` (the only implementation through Phase 1-4) happens to run stages synchronously - means the Phase 9 hardening cutover to `CeleryTaskRunner` is a configuration change (`TASK_RUNNER=celery`) plus implementing the currently-stubbed class, not a rewrite of every pipeline call site.

## Why does `CeleryTaskRunner` raise `NotImplementedError` instead of being implemented now?

Because there is no pipeline stage yet to dispatch - implementing Celery integration against nothing would be exactly the kind of unused placeholder this project's rules forbid. The class's docstring documents the intended design precisely so the seam is visible and the eventual implementation isn't invented from scratch:

- `dispatch()` enqueues a Celery task and returns immediately - it must never block an HTTP request waiting for ML work that can take seconds to minutes.
- Celery tasks are plain `def` functions; each one runs `asyncio.run(stage.run(input))` internally and persists status/output into a `pipeline_runs` table, not into Celery's result backend (which is reserved for a small completion marker only, avoiding size/serialization mismatches with real pipeline output).
- `get_status()`/`get_result()` read from `pipeline_runs`, so status survives worker restarts and doesn't depend on Celery/Redis being reachable at query time.
- Low-latency UI updates (the Phase 5 streaming voice experience) additionally use Redis pub/sub for task completion, rather than client-side polling alone.

## Test coverage

`apps/api/tests/unit/test_task_runner.py` covers `InProcessTaskRunner`'s success and failure paths, and asserts `CeleryTaskRunner` fails loudly (raises `NotImplementedError`) rather than silently pretending to run a job.
