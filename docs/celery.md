# Background jobs: Celery (Phase 9, hardened in Phase 9.1)

`TASK_RUNNER` selects between two real implementations of the same `TaskRunner` contract established in Phase 1 (`workers/task_runner.py`, ADR 0002) - `dispatch(stage, input) -> JobHandle` returns immediately; callers poll `get_status()`/fetch `get_result()`:

- `in_process` (default) - runs a stage synchronously in the caller's event loop. No Redis, no worker process. This is everything Phases 1-8 used.
- `celery` - enqueues a real Celery task through a real Redis broker; a separate worker process executes it and persists status/output to Postgres. This is what Phase 9 adds.

**No pipeline logic was duplicated to build this.** Every stage's actual `run()` method is untouched; Celery only had to solve "how does a worker process reconstruct the right stage instance from a message that crossed a broker."

## How a stage actually gets from an HTTP request to a worker and back

```
HTTP request
  -> service layer (e.g. KnowledgeService) checks ownership, constructs a
     real stage instance with real dependencies (exactly as it always has)
  -> task_runner.dispatch(stage, input)
       - CeleryTaskRunner: persists a `pipeline_runs` row (status=pending)
         in Postgres, then calls celery_app.send_task(..., args=[job_id]) -
         only the job_id (a UUID string) crosses the broker, never the
         stage object, never the real payload, never a secret
       - returns a JobHandle immediately - the HTTP request is never
         blocked waiting for the real work
  -> (client polls) task_runner.get_status(job_id) / get_result(job_id)
       - both read straight from the `pipeline_runs` Postgres row -
         independent of whether Redis/the worker is reachable right now

Meanwhile, in a separate worker process:
  celery worker picks up the message
  -> plain `def execute_pipeline_stage(self, job_id)` (Celery cannot invoke
     `async def` - this is the one required sync/async boundary)
  -> asyncio.run(_execute_pipeline_stage_async(self, job_id))
       - opens its own worker-local Postgres session (never the API
         process's request-scoped session - see workers/db.py)
       - reads the `pipeline_runs` row for stage_name + input_json
       - looks up stage_name in workers/stage_registry.STAGE_REGISTRY,
         which reconstructs a fresh, real stage instance from Settings
         alone (the same dependency-construction code
         voxmind.api.deps already uses) - not a duplicate of any stage's
         logic, just its wiring
       - deserializes input_json into the real Pydantic input class and
         calls the stage's real, unmodified `run()`
       - persists status=completed/failed + output_json back to the same
         Postgres row
```

### Why the stage object itself never crosses the broker

Every `PipelineStage`'s constructor dependencies (`StorageBackend`, `FasterWhisperProvider`, etc.) are cheap to reconstruct from `Settings` alone - there's no live, request-specific state inside them. Pickling the actual Python object across Redis instead would mean using Celery's `pickle` serializer (a real remote-code-execution risk for a message broker) and would tie every message's size/shape to internal implementation details. Reconstructing by name (`stage.name`, already a required `PipelineStage` attribute) from a small, versioned registry is simpler and strictly safer - see `workers/stage_registry.py`'s module docstring.

## Running it for real

```bash
# 1. Redis (broker + result backend)
brew install redis && brew services start redis
# or: docker compose -f docker/docker-compose.yml up -d redis

# 2. In apps/api/.env:
TASK_RUNNER=celery
REDIS_URL=redis://localhost:6379/0
REDIS_RESULT_BACKEND_URL=redis://localhost:6379/1

# 3. Start a worker (separate terminal, from apps/api/ with the venv active)
celery -A voxmind.workers.celery_app:celery_app worker \
  --loglevel=info --queues=voxmind.ml,voxmind.default --concurrency=2

# 4. Start the API as usual
uvicorn voxmind.main:app --reload
```

`GET /api/v1/health` and `GET /api/v1/health/ready` report real, live reachability of Redis and a real worker (via `celery_app.control.ping()`, not just "is the broker port open") - see their docstrings in `api/v1/endpoints/health.py`.

## Queues

Two queues, not one per stage - the "appropriate task boundary" is one message per `PipelineStage.run()` call (matching what `InProcessTaskRunner.dispatch()` already did per call), and unnecessary per-tiny-operation queues would just add routing complexity with no real benefit:

| Queue | Stages | Why separate |
|---|---|---|
| `voxmind.ml` | whisper_transcription, speaker_diarization, acoustic_feature_extraction, wav2vec2_embedding, emotion_inference, nlp_analysis, chunk_embedding, rerank, tts_synthesis | Model-loading, CPU/memory-heavy - an operator can scale or isolate this worker pool independently |
| `voxmind.default` | audio_preprocessing, transcript_alignment, document_ingestion, incongruence_analysis | Cheap, deterministic or I/O-bound - never worth a dedicated pool |

`workers/stage_registry.py::queue_for_stage()` is the single place this routing decision is made.

## Reliability

- **`task_acks_late=True` + `worker_prefetch_multiplier=1`**: a task is only acknowledged after it finishes, so a worker killed mid-task (deploy, OOM, crash) leaves the message requeued for another worker instead of silently losing the job; prefetch=1 stops one worker hoarding several minutes-long ML tasks while sibling workers idle.
- **`broker_transport_options={"visibility_timeout": 2 * CELERY_TASK_TIME_LIMIT_SECONDS}`**: Kombu's Redis transport (unlike AMQP) decides a delivered-but-unacked message is eligible for redelivery purely by *time*, not by detecting a dropped connection. Left at Kombu's own default (3600s), a truly crashed worker's task would sit unrecovered for up to an hour. Set explicitly, proportionate to this app's actual task durations - long enough that a still-legitimately-running task is never mistaken for lost and redelivered out from under itself, short enough that a genuine crash recovers in minutes. `task_reject_on_worker_lost` is deliberately left at Celery's default (`False`) - see `workers/celery_app.py`'s module docstring for the full reasoning.
- **Idempotency / redelivery guard**: the worker checks `pipeline_runs.status` before doing any work - a row already in a terminal state (`completed`/`failed`) is treated as a no-op, proven (not just asserted) by making the stage's own factory raise if it's ever reached for a terminal row. `failed` is exactly as terminal as `completed` - a run this project's own retry logic already gave up on is never silently resurrected into `completed` by an unrelated redelivery, even if the underlying condition has since resolved (`test_redelivery_of_an_already_failed_run_never_becomes_success`). A row still in `running` (a genuine worker-loss scenario - the original attempt never reached a terminal state) is *not* a no-op and correctly re-executes to completion, since every real `PipelineStage.run()` in this codebase is side-effect-idempotent (storage writes go to the same deterministic key; no stage writes its own DB rows).
- **Retry policy**: only genuinely transient failures (`TimeoutError`, `ConnectionError`, `OSError`) are retried, with exponential backoff (`CELERY_TASK_RETRY_BACKOFF_SECONDS * 2**retries`), up to `CELERY_TASK_MAX_RETRIES` (default 3, and genuinely bounded - `test_transient_failures_stop_retrying_once_max_retries_is_reached` proves a transient error still fails permanently once the budget is exhausted). Everything else - a stage's own domain exceptions (`AudioProcessingError`, `UnsupportedAudioFormatError`), a missing/invalid credential, a Pydantic validation error on the input, an unknown stage name - fails immediately and permanently. Retrying those would never succeed and would just burn worker time.
- **Timeouts reach a real terminal state, not a stuck `running` row** - the one area a Phase 9.1 hardening pass genuinely found and fixed a live bug in (see "A real bug this hardening pass found" below). `CELERY_TASK_SOFT_TIME_LIMIT_SECONDS` (270s) fires first and is always classified permanent (never retried); `CELERY_TASK_TIME_LIMIT_SECONDS` (300s) is a 30-second backstop.
- **No orphaned rows**: `dispatch()` marks the `pipeline_runs` row failed immediately if enqueueing itself throws (broker unreachable) - a row is never left stuck in `pending` forever with nothing that will ever pick it up.
- **Serializers are `json` only, never `pickle`**: enforced in `workers/celery_app.py`'s config and guarded by a regression test (`tests/unit/test_celery_config.py`).

### A real bug this hardening pass found (and fixed)

`billiard`'s soft-time-limit signal fires wherever process execution happens to be when the timer expires. For any stage blocked in real I/O (`asyncio.sleep`, a socket read, Whisper's `asyncio.to_thread()` call - i.e. almost every real stage in this pipeline), that's *inside asyncio's own event-loop machinery* (its internal `selector.select()` call), not inside the awaited coroutine's own frame. The exception therefore propagates straight out of `asyncio.run()` instead of ever reaching the `except SoftTimeLimitExceeded` clause inside `_execute_pipeline_stage_async`.

This was caught for real, live, by `test_real_soft_time_limit_kills_a_genuinely_slow_stage` (a genuinely slow test-only stage, a real per-message `soft_time_limit`/`time_limit` override, a real worker subprocess): the naive single-try version of `execute_pipeline_stage` left the `pipeline_runs` row stuck at `running` forever. The fix wraps the entire `asyncio.run(...)` call in its own `try/except SoftTimeLimitExceeded`, using a *fresh* `asyncio.run()` (the original loop is gone) to record the real outcome - see `workers/tasks.py`'s module docstring for the full explanation and both code paths.

A second, unrelated real bug surfaced by the same hardening pass, via `test_real_worker_runs_tts_synthesis`: `output.model_dump(mode="json")` raises `UnicodeDecodeError` on a real `bytes` field containing genuine binary data (`TtsResult.audio_bytes` - a real synthesized WAV is not valid UTF-8), and that call used to sit *outside* the task's try/except entirely, so the failure escaped uncaught and again left the row stuck at `running`. Fixed in `workers/json_safe.py` (base64-wraps `bytes` fields for JSONB storage, reversed on read - never changes `TtsResult`/`TtsSynthesisStage` themselves, which are unrelated, already-working Phase 5 code) and by moving the serialization call inside the try block.

### Known, honest, NOT fixed limitations

- **The hard time limit remains a real gap**: if it ever fires (a backstop for code that isn't interruptible by the soft limit's signal at all - e.g. stuck in a non-yielding C call), the OS kills the worker child process directly and no Python code runs at all, so the `pipeline_runs` row is left at `running`. The same is true of a genuine `asyncio.CancelledError`/`SystemExit`/`KeyboardInterrupt` reaching this code (all `BaseException`, not `Exception`, so neither of the two `SoftTimeLimitExceeded` handlers catches them) - this project deliberately does not catch bare `BaseException` to paper over this, since doing so risks masking a legitimate process-shutdown signal. Both are the same underlying category of gap: a signal/cancellation that bypasses Python-level cleanup entirely. **This is exactly the gap reconciliation (below) recovers from** - a row left stuck at `running` by either case is picked up and terminated once it crosses the reconciliation threshold, even though nothing caught the original failure.
- **Poison-pill redelivery - now bounded (final hardening pass)**: see "Poison-pill bound" below.

## Poison-pill bound (final hardening pass)

The gap above was real: if a specific input reliably crashed whatever worker picked it up, Celery's Redis transport would keep redelivering it every `visibility_timeout` seconds indefinitely - `CELERY_TASK_MAX_RETRIES`/`retry_count` only ever governed *caught, in-process* retries (a transient exception the task body itself classifies and hands to `task.retry()`), never broker-level redelivery after a worker is lost entirely (crashed, OOM-killed, hard-timed-out) - that path never touches `retry_count` at all, since it isn't a retry the task body ever gets a chance to observe.

**The fix**: `PipelineRun.delivery_count`, a new column incremented once in `PipelineRunRepository.mark_running()` - the one place, in production, every genuine execution attempt passes through identically, whether it's the first dispatch, an in-process retry, or a broker redelivery after worker loss. `workers/tasks.py::_execute_pipeline_stage_async` checks it before doing any real work: once a row has reached `CELERY_MAX_DELIVERY_ATTEMPTS` (10, deliberately well above the *normal* retry path's own maximum of `1 + CELERY_TASK_MAX_RETRIES = 4` attempts, giving real headroom for a few genuine crash-redelivery cycles before concluding the input is a true poison pill), the run is marked `failed` with a clearly-labeled error and the stage is never invoked again - closing the loop on uncontrolled, repeated *expensive* processing (the real harm - unbounded broker redelivery of the bare message itself was never actually infinite in practice, since `visibility_timeout` already paces it).

This is deliberately a **database-backed attempt guard**, not a new broker-level mechanism or a second queueing system - reusing exactly the same `pipeline_runs` table and `PipelineRunRepository` every other piece of this project's reliability story already goes through. It composes cleanly with everything already in place:

- **Normal, bounded in-process retries** (a transient `ConnectionError`/`TimeoutError`/`OSError`, retried up to `CELERY_TASK_MAX_RETRIES` times) are unaffected - they exhaust their own, smaller, unchanged budget first in the overwhelming majority of real cases.
- **Worker-loss redelivery** (the case `retry_count` never saw) is now bounded too, since every redelivered re-execution still passes through `mark_running()`.
- **Duplicate/late delivery** after a run is already terminal remains a pure no-op via the pre-existing idempotency guard, completely unaffected by this addition - `delivery_count` is only ever consulted for rows still in a non-terminal state.
- **Reconciliation** remains unaffected and complementary, not overlapping: it recovers a row that's been *abandoned* (nothing has touched it in a long time, measured by `started_at`); this bound recovers a row that's being *actively, repeatedly, unsuccessfully* redelivered (measured by `delivery_count`). A row could in principle hit either terminal path first, depending on which condition it satisfies first - both leave it correctly `failed`, never in conflict.

See `tests/integration/test_celery_task_runner.py`'s `test_a_run_that_already_exhausted_its_delivery_budget_fails_without_executing` (a stage that would raise `AssertionError` if actually invoked, proving the exhausted-budget path never reaches it) and `test_delivery_count_increments_across_repeated_in_process_retries_and_bounds_them` (a real, deterministic multi-delivery simulation proving the counter climbs and the bound is eventually hit) for the real test coverage.

## Worker orphan/fencing (final hardening pass)

A real, narrow gap surfaced while building the earlier hard-worker-kill test for reconciliation (see "Hard worker failure" above): killing only a worker's *top-level* process, rather than its whole container/cgroup, can leave an orphaned forked child still executing. That orphan doesn't know its parent is gone, keeps running the real stage to completion, and eventually tries to persist a result - potentially *after* reconciliation (or another worker's redelivery-driven re-execution) has already moved the row on. Without a check, that stale write would silently apply, resurrecting or overwriting a decision something else already made.

**The fix**: a real, database-level compare-and-swap, using the row's own existing `status` + `started_at` pair as the fencing token - no new column, no new locking system, no distributed coordination service. `_execute_pipeline_stage_async` captures `expected_started_at = run.started_at` immediately after its own `mark_running()` call succeeds - the exact value *this* execution attempt itself just set. Every terminal write this same attempt later makes (`PipelineRunRepository.mark_completed_if_still_owned()`, `mark_failed_if_still_owned()`, and `reset_to_pending_for_retry_if_still_owned()` for the in-process-retry path) is a conditional `UPDATE ... WHERE id = :id AND status = 'running' AND started_at = :expected_started_at` - it only actually applies if nothing else has touched the row since. If the write's `rowcount` comes back `0`, the attempt is treated as `"superseded"`: logged, returned honestly, and the caller's in-memory result is discarded rather than resurrecting a row this attempt no longer owns.

This composes correctly with everything else here without any extra locking on the worker's side: a plain conditional `UPDATE` from a stale worker and reconciliation's own `SELECT ... FOR UPDATE SKIP LOCKED` (see above) naturally serialize against each other through ordinary Postgres row-level locking - whichever transaction commits first "wins," and the other's `WHERE` clause then correctly matches zero rows once it evaluates against the new, already-committed state.

Verified for real, not just asserted by construction: `tests/integration/test_celery_task_runner.py::test_a_stale_workers_completion_is_rejected_after_reconciliation_supersedes_it` has a real stage `run()` reach into the database through a *genuinely separate connection* (`worker_session_scope()`, not the session the task itself is using) to actually commit a reconciliation-style write while the task is still "mid-execution" from its own perspective, then confirms the task's own subsequent completion write is rejected and the reconciled terminal state is untouched. `test_a_normal_active_worker_still_completes_successfully_with_fencing_in_place` confirms the ordinary, unaffected case still works.

## Reconciliation (final verification pass)

`workers/reconciliation.py::reconcile_stale_pipeline_runs()` closes the "no periodic reconciliation sweep" gap noted in earlier phases: it finds every `pipeline_runs` row stuck at `status="running"` whose `started_at` is older than `CELERY_RECONCILIATION_STALE_THRESHOLD_SECONDS` (1200s by default - derived, not guessed, from `visibility_timeout` + one full redelivered execution attempt + a safety margin; see the module docstring and [DECISIONS/0017](DECISIONS/0017-pipeline-run-reconciliation.md) for the full derivation) and marks each one `failed` with a clearly-labeled, greppable error (`"reconciliation: ..."`).

It is a plain function operating purely on Postgres, invoked via `python -m voxmind.workers.reconciliation` (a thin CLI entrypoint) - the same function a human can still run by hand after a known incident, and, since the final limitations-clearance pass below, the same function a real Celery Beat schedule now also invokes automatically. See [DECISIONS/0017](DECISIONS/0017-pipeline-run-reconciliation.md) for the original "why not Beat" reasoning and its later reversal. It never touches `pipeline_runs` over HTTP - same guard (`tests/unit/test_pipeline_run_not_exposed.py`) as everything else in this table.

It is idempotent (a reconciled row is terminal and never matches the `running`-only filter again), never touches a `pending`/`completed`/already-`failed` row, and never touches a `running` row within the threshold - including one whose `started_at` keeps being refreshed by real redelivery, which correctly never looks stale. If the original broker message somehow still exists and is redelivered after reconciliation already gave up on a row, `workers/tasks.py`'s pre-existing "failed is as terminal as completed" idempotency guard prevents it from ever being resurrected - reconciliation inherits this protection for free rather than reimplementing it. See `tests/unit/test_reconciliation.py` for the full scenario coverage.

### Automatic scheduling (final limitations-clearance pass)

Reconciliation now runs on a real Celery Beat schedule, closing the "needs external scheduling to actually run" limitation the original design (above) deliberately left open. `celery beat` is a scheduling *mode* already built into the `celery` package this project has depended on since Phase 9 - enabling it via `celery worker -B` (`docker-compose.full.yml`) is reuse of existing infrastructure, not a new product or dependency, and it runs embedded in the same single worker process rather than as a separate container (this project runs exactly one worker; a dedicated `beat` service would only earn its keep once there is more than one). `workers/tasks.py::reconcile_stale_pipeline_runs_task` is a thin periodic-task wrapper - the CLI entrypoint and the scheduled task both call the identical `reconcile_stale_pipeline_runs()` function, unchanged.

`CELERY_RECONCILIATION_INTERVAL_SECONDS` (300s default) controls how often the check itself runs - independent of, and always shorter than, `CELERY_RECONCILIATION_STALE_THRESHOLD_SECONDS` (how patient it is before calling a row abandoned). The query now also uses `FOR UPDATE SKIP LOCKED`, real database-level protection in case a slow pass is still in flight when the next tick fires (or, in principle, more than one scheduler runs) - proven for real by `test_two_concurrent_reconciliation_passes_never_double_process_the_same_row`, which runs two genuinely independent Postgres sessions concurrently and confirms every stale row is reconciled by exactly one of them, never both, never neither.

**A real scheduled execution was verified against the actual Docker/Celery environment**, not just unit-tested: with a temporary, test-only short interval/threshold (10s/5s, reverted immediately after), a stale row inserted directly into the real containerized Postgres was picked up and reconciled by a genuine `Beat: Scheduler: Sending due task` tick within one interval, confirmed by the row's real `status`/`error`/`completed_at` afterward. Repeated across six consecutive real ticks: correctly a no-op (`reconciled_count=0`) whenever nothing was stale, and correctly catching each newly-stale row exactly once. Not a simulation - `docker logs`/`psql` against the live containers.

**A genuine hard-worker-kill recovery was also verified for real** (`docker kill -s KILL` against the live `worker` container mid-task, a real restart, a real Beat tick reconciling the abandoned row, a real simulated late redelivery correctly treated as a no-op) - see `tests/integration/test_celery_task_runner.py::test_a_hard_killed_worker_leaves_a_stuck_run_that_reconciliation_recovers` for the equivalent, CI-portable local version. **A real, narrow finding surfaced while building that test**: killing only a worker's top-level process (rather than its whole process tree/container/cgroup) can leave an orphaned prefork child still executing independently, since `task_acks_late`'s idempotency guard is only checked once, at the *start* of `_execute_pipeline_stage_async` - a child already past that check will finish and write its result regardless of what reconciliation does to the row in the meantime, in principle re-completing a run reconciliation already closed out. This is a real, inherent limitation of any purely database-driven reconciliation approach without an additional liveness signal (e.g. checking Celery's own `control.inspect` before declaring a row abandoned), not something this pass fixes - genuine "whole worker gone" failures (a container SIGKILL, an OOM-killer tearing down a full cgroup - the realistic hard-failure shapes this project actually verified live) kill the entire process tree together and are unaffected, so this narrows to the less common case of a partial kill (e.g. an operator signaling only the parent PID). Documented honestly rather than silently assumed away.

## Worker configuration

The `prefork` pool (Celery's default) is the right starting point for this workload - it's CPU-heavy (Whisper, Wav2Vec2, sentence-transformers, the reranker), and prefork gives real process isolation so one worker crashing on a bad model input doesn't take down its siblings. Concurrency is deliberately modest (2 in the example above, 2 in `docker-compose.full.yml`) because each forked child can load its own copy of a multi-hundred-MB model into memory. `--max-tasks-per-child` is set in the Docker Compose worker service to recycle child processes periodically and bound any slow memory growth from repeated model loads.

### Memory robustness - real, measured findings (final verification pass)

This wasn't left as an assumption. A real, local stress run (native Postgres + Redis, a real `celery worker` subprocess with production's own `--concurrency=2`, dispatching real sequential tasks through `CeleryTaskRunner` - not mocked, not simulated) measured actual worker-process-tree RSS via `ps` before, during, and after:

- **20 real sequential `chunk_embedding` tasks (one real HuggingFace sentence-embedding model), no recycling limit**: RSS started at 637MB (cold import of torch/transformers before any model finishes loading), then **plateaued at 354-371MB from task 4 onward** - no growth trend across the remaining 16 tasks. Final RSS after completion was 301MB, below the starting baseline.
- **20 real sequential tasks alternating between two different real ML models (`chunk_embedding` and `rerank`'s cross-encoder), no recycling limit**: same pattern - RSS settled to a **360-368MB plateau** by task 8 and stayed flat through task 19, confirming this isn't an artifact of calling one model repeatedly; loading two different real models into the same long-lived worker process doesn't produce unbounded growth either.
- **15 real sequential tasks with `--max-tasks-per-child=5`** (an aggressive recycling threshold, well below production's 50): RSS climbed to a **968MB peak** - measurably *worse* than the no-limit runs above. The worker log confirms recycling genuinely happened (`ForkPoolWorker-1` -> `ForkPoolWorker-3` -> `ForkPoolWorker-2` -> `ForkPoolWorker-4`, new child identities appearing as old ones hit the cap), so this isn't a broken recycling mechanism - it's real evidence that, for this codebase's actual ML stages, repeatedly cold-starting fresh child processes (re-importing `torch`/`transformers`/`sentence-transformers` from scratch, which itself costs real, non-trivial memory before any model weight loads) is **more expensive than any leak it would prevent**, at least at a low threshold and over a short run.

**Conclusion, and why nothing was changed here**: no unbounded per-task memory growth was found in this codebase's real stages once a model is warm - the existing `@lru_cache`-based provider construction (`workers/stage_registry.py`) already keeps a worker child from reloading a model on every task, which is the real reason memory plateaus rather than climbing. Production's existing `--max-tasks-per-child=50` (`docker-compose.full.yml`) is **kept unchanged** - the real evidence above says lowering it further would hurt, not help, and there was no observed leak to justify a stricter (or an additional, separate `--max-memory-per-child`) limit either. Per-worker peak RSS in these real measurements topped out under 1GB even in the deliberately-adversarial low-threshold run, comfortably inside the project's known 3.8GB Docker VM budget (see `docs/docker.md`) alongside Postgres/Redis/MinIO/the API process. This is a real, evidence-based decision, not an arbitrary value chosen to declare the task "done" - if a genuine slow leak is ever observed in production (a metric this document does not have a live production system to gather), revisiting this with real numbers from that environment is the right next step, not preemptively guessing a limit now.

`tests/integration/test_celery_task_runner.py::test_real_worker_recycles_child_processes_without_breaking_task_correctness` is the permanent regression coverage for this: it forces several real recycle events (`--max-tasks-per-child=2`, deliberately more aggressive than production, to make the test fast and deterministic) within one real worker subprocess and asserts every dispatched task still completes correctly across each recycle boundary - proving the recycling mechanism itself never corrupts a task's result, independent of the memory-tuning question above.

## Observability

Every task logs (via the existing structlog setup, never a separate logging system) `job_id`, `celery_task_id`, `stage_name`, `queue`, `correlation_id`, `retries`, `worker_hostname`, `status`, `duration_ms`, and the final outcome/error. `correlation_id` is captured automatically from `structlog.contextvars` at `dispatch()` time (the same `request_id` `RequestContextMiddleware` has bound to every HTTP request since Phase 5) - a worker-process log line can be traced back to the exact API request that triggered it, without `TaskRunner.dispatch()`'s signature ever needing to change (verified by `test_dispatch_captures_the_real_request_correlation_id`). Nothing sensitive (tokens, passwords, API keys, raw audio/prompt content) is ever logged - the same discipline this project has applied since Phase 1.

## Security

- **`pipeline_runs` is never reachable over HTTP** - it's an internal `TaskRunner` implementation detail. User-facing job tracking uses separate, already-ownership-checked tables (`AudioProcessingJob`, `EmotionProcessingJob`, `VoiceTurn`, ...). There is therefore no ID-guessing/cross-user-exposure surface for a `pipeline_runs.id` to defend in the first place - locked in by `tests/unit/test_pipeline_run_not_exposed.py`, which fails if any endpoint module ever imports `PipelineRun` or any route path ever mentions "pipeline".
- **Only a UUID crosses the broker**, regardless of how large the real input is - `test_large_input_never_crosses_the_broker_only_a_job_id_does` dispatches a 200KB payload and confirms `send_task`'s arguments stay under 100 bytes.
- **Deserialization is Pydantic validation, never `pickle`** - `registration.input_model.model_validate(run.input_json)` and the `json`-only serializer configuration (see above) rule out arbitrary code execution via a crafted broker message.
- **`/health` and `/health/ready` are deliberately unauthenticated** (matching the pre-Phase-9 `/health` endpoint, and standard practice for infra probes a load balancer/orchestrator must call without credentials) and do disclose real infrastructure reachability (Redis/worker status) to an unauthenticated caller - a reviewed, accepted, industry-standard tradeoff, not an oversight.

## Testing

- `tests/unit/test_celery_config.py` - deterministic checks on the real Celery app's configuration (serializers, `acks_late`, `visibility_timeout`, queue routing) and that every real `PipelineStage` class in the codebase is registered - no broker needed.
- `tests/unit/test_reconciliation.py` - real Postgres-backed coverage of `reconcile_stale_pipeline_runs()`: every status left untouched, a genuinely stale run reconciled, idempotency across repeated calls, a long-running legitimate task (refreshed by simulated redelivery) never prematurely reconciled, and - through the real worker task body, not just the repository layer - a reconciled run never resurrected by a later redelivery. No broker needed.
- `tests/unit/test_json_safe.py` - deterministic regression coverage for the real `bytes`-serialization bug described above, using fixture binary data (not real audio).
- `tests/unit/test_pipeline_run_not_exposed.py` - the security property above, checked structurally (AST inspection of endpoint modules + a real FastAPI route-table scan).
- `tests/integration/test_celery_task_runner.py` - real Postgres-backed tests of `CeleryTaskRunner` and the worker task body: status transitions, real output persistence, retry classification (including retries genuinely stopping once exhausted), the redelivery/idempotency guard (including the "failed never becomes completed" and "running correctly re-executes" cases), and a deterministic simulation of a soft-time-limit timeout reaching a real terminal state. Six tests are marked `requires_redis` and start a genuine Celery worker subprocess against a genuine Redis broker - `test_real_broker_round_trip`, `test_real_soft_time_limit_kills_a_genuinely_slow_stage`, and one real, live, credential-free representative stage from each remaining category this pipeline has (`test_real_worker_runs_document_ingestion`, `_audio_preprocessing`, `_rerank`, `_tts_synthesis`) - skipped, never faked, when no broker is reachable in the current environment.

## What Phase 9 deliberately did not change

The `PipelineStage.run(input) -> output` and `TaskRunner.dispatch()/get_status()/get_result()` contracts are byte-for-byte the same as Phase 1. `InProcessTaskRunner` is untouched and remains the default; switching to Celery is a configuration change (`TASK_RUNNER=celery`) plus running a worker, not a redesign anywhere else.

**Update (Phase 10 final validation) - this claim used to also say no service call site ever needed to change. That turned out to be wrong**, and is corrected here rather than left inaccurate: every real call site across `AudioService`, `EmotionService`, `NlpService`, `IncongruenceService`, `KnowledgeService`, and `RetrievalService` checked `get_status()` exactly once immediately after `dispatch()`, which is silently correct for `InProcessTaskRunner` (already-synchronous) but is a real race against `CeleryTaskRunner`, whose `dispatch()` only enqueues and returns before any worker has run. This was never exercised by CI or local dev (both default to `TASK_RUNNER=in_process`), and the dedicated Celery tests below exercise `CeleryTaskRunner` and the worker task body directly rather than a real HTTP request through these actual services with `TASK_RUNNER=celery`. Found live during final validation (a real `500` on `POST /conversations/{id}/audio/{asset_id}/process` against a real Celery worker) and fixed by adding `workers/task_runner.py::wait_for_completion()`, a bounded polling helper every one of those twelve call sites now uses instead of a single check - see [DECISIONS/0011](DECISIONS/0011-celery-dispatch-poll-race.md) for the full account and live before/after verification.

**Also observed during final validation, honestly noted rather than chased further**: `tests/integration/test_celery_task_runner.py`'s `requires_redis`-marked tests each pass individually, but running the whole file back-to-back on this project's 8GB-memory development machine is flaky - several of these tests spawn their own real `celery worker` subprocess that loads a genuine multi-hundred-MB ML model (the reranker, the local TTS model, emotion2vec+), and enough of those in sequence under this machine's real memory pressure can leave a later subprocess too slow to reach a terminal state inside its test's fixed deadline. This is a real, environment-specific resource constraint of the test file's own design (independently confirmed: `vm_stat` showed the machine under genuine memory pressure - free pages in the low thousands, heavy compressor/swap activity - while this was reproduced), not a defect in `CeleryTaskRunner`, the worker, or any call site - the same real production Celery path (one worker, real HTTP requests, `TASK_RUNNER=celery`) was independently verified working end-to-end for both audio processing and emotion inference, see [DECISIONS/0011](DECISIONS/0011-celery-dispatch-poll-race.md) again. Not fixed here, since doing so would mean reworking this test file's own subprocess-per-test design - out of scope for a validation pass whose job is fixing real integration bugs, not rewriting an already-working test suite for a machine-specific resource constraint.
