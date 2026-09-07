# ADR 0017: pipeline_runs reconciliation - a plain function + CLI entrypoint, not Celery beat

**Update (final limitations-clearance pass): superseded on the scheduling question.** The "not Celery beat" decision below is preserved verbatim for the record - it was the right call at the time, and the reasoning (avoid new infrastructure) remains sound in the abstract. It has since been revisited and reversed: `celery beat` was judged to be a scheduling *mode* of the `celery` package this project already depends on, not a new product, and the remaining "needs external scheduling to actually run" limitation this ADR knowingly left open was worth closing now that this project needed it for real. See `workers/reconciliation.py`'s module docstring ("Scheduling, and why it's Celery Beat now") and `docs/celery.md`'s "Automatic scheduling" section for the current design, including a real, verified scheduled execution against the live Docker/Celery environment. `reconcile_stale_pipeline_runs()`'s own contract (below) is completely unchanged by this - the periodic task is a thin wrapper around the exact same function the CLI entrypoint already called.

## The gap this closes

`task_acks_late=True` + `broker_transport_options["visibility_timeout"]` (see `workers/celery_app.py`'s docstring) already give this project real, working self-healing for the *common* crash case: a worker killed mid-task leaves its message unacked, Redis redelivers it after `visibility_timeout` seconds, a fresh worker re-executes the same `job_id`, and the existing idempotency guard in `workers/tasks.py` makes that safe. Nothing new was needed for that path.

The real, honest gap - flagged since Phase 9.1 in `celery_app.py`'s module docstring and `docs/celery.md`'s "Known, honest, NOT fixed limitations" - is the case where redelivery *doesn't* happen at all: the broker message is genuinely gone (Redis restarted or flushed between the crash and the next `visibility_timeout` boundary, or some other operational action removed it), while the `pipeline_runs` row is left at `status="running"` forever, since nothing in this codebase ever revisited it. This ADR is that fix.

## The choice: a plain, idempotent function + CLI entrypoint, not Celery beat

Adding `celery beat` would mean running a genuinely new, long-lived scheduler process (with its own schedule/state file) this project doesn't run anywhere today - not in `docker-compose.full.yml`, not in local dev. That's real new infrastructure, not a reuse of what already exists, and the project's own instruction for this pass was explicitly to prefer the smallest production-sensible solution and avoid any external distributed system beyond existing infra.

`workers/reconciliation.py::reconcile_stale_pipeline_runs()` is a plain async function operating purely on Postgres - no broker, no worker subprocess needed to run or test it. `python -m voxmind.workers.reconciliation` is a thin CLI entrypoint around it, intended to be invoked by whatever an operator already has available: a host cron entry, a Docker/Kubernetes scheduled job, or manually after a known incident. It is safe to run repeatedly (idempotent - see below) and safe to run on a schedule this project doesn't itself need to define. Adopting a real periodic scheduler later (Celery beat or otherwise) would call this exact function unchanged.

## Why the threshold is derived, not guessed

`CELERY_RECONCILIATION_STALE_THRESHOLD_SECONDS` (`core/config.py`, default 1200s) has to clear two real waiting periods before a `running` row can be safely called abandoned:

1. `broker_transport_options["visibility_timeout"]` (2 x `CELERY_TASK_TIME_LIMIT_SECONDS` = 600s by default) - how long Redis waits before a delivered-but-unacked message becomes eligible for redelivery at all.
2. One full redelivered execution attempt (another `CELERY_TASK_TIME_LIMIT_SECONDS` = 300s) - the time a freshly redelivered task is allowed to legitimately run before Celery's own hard limit would kill it.

600s + 300s = 900s is the earliest a row could still be legitimately mid-recovery; 1200s adds a real safety margin on top rather than cutting it exactly at the theoretical minimum. Critically, `workers/tasks.py` calls `mark_running()` (refreshing `started_at`) on *every* execution attempt, including ones triggered by redelivery - so a row that's still genuinely cycling through Celery's own recovery mechanism keeps its `started_at` fresh and never crosses this threshold. Only a row nothing has touched across this entire window is a real candidate - see the module docstring for the full argument.

## Why `status="failed"`, not a new fifth status

`PipelineRun.status`'s `CHECK` constraint (`models/pipeline_run.py`) has only ever allowed `pending`/`running`/`completed`/`failed` - the same four values `workers.task_runner.JobStatus` has used since Phase 9. Adding a distinct `cancelled`/`reconciled` status would require a schema migration and new handling in `CeleryTaskRunner.get_status()` and every caller that maps `JobStatus` today, for no real behavioral gain - every caller today only ever distinguishes terminal-failed from everything else. Reconciliation marks a row `failed` with a clearly-labeled, greppable error prefix (`"reconciliation: ..."`) instead - the same "one well-understood mechanism, not a second overlapping one" reasoning `docs/celery.md`'s reliability section already applies to `task_reject_on_worker_lost`.

## Why this needed no HTTP surface, and doesn't get one

`pipeline_runs` is already, deliberately, never reachable over HTTP (`tests/unit/test_pipeline_run_not_exposed.py`). Reconciliation doesn't change that: `workers/reconciliation.py` lives entirely outside `voxmind/api/`, adds no route, and is only ever invoked as a plain function (from a test) or via its own CLI entrypoint (from an operator's shell/scheduler) - never from a request handler.

## Idempotency and safety, proven by tests

`tests/unit/test_reconciliation.py` covers every scenario this pass required: a normal completed run is untouched; an active running run within the threshold is untouched; a genuinely stale running run is reconciled; repeated reconciliation calls on the same row are no-ops after the first (idempotent); a failed run (including one representing a deliberate cancellation, since this schema has no separate status for that) is untouched; a pending run is untouched; a long-running *legitimate* task whose `started_at` keeps being refreshed by real redelivery is never prematurely reconciled; and - driven through the real worker task body, not just the repository layer - a row reconciliation already marked `failed` is never resurrected into `completed` even if the original message is somehow still redelivered afterward (inherited for free from the pre-existing "failed is exactly as terminal as completed" guard in `workers/tasks.py`).

## A real, honest limitation this doesn't solve

Reconciliation has no scheduler of its own - if nobody ever invokes `python -m voxmind.workers.reconciliation` (or wires it into cron/a scheduled job), stale rows are correctly *identifiable* but never automatically corrected. This is a deliberate, disclosed consequence of choosing not to add a new always-on scheduler process as part of this pass, not an oversight - see "The choice" above.
