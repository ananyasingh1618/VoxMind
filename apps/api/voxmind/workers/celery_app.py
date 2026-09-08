"""The real Celery application (Phase 9) - Redis broker, Redis result
backend reserved for Celery's own small completion marker only (see
`workers/tasks.py` and ADR 0002; real pipeline output lives in
`pipeline_runs`, never in the Celery result backend).

`task_serializer`/`result_serializer`/`accept_content` are pinned to
`"json"`, never `"pickle"` - a message broker that can deserialize arbitrary
pickled Python objects is a real remote-code-execution risk if it's ever
reachable by anything other than this application's own trusted workers.
Every task payload is therefore just a `job_id` string (see
`CeleryTaskRunner.dispatch()`), never a large blob and never a secret.

`task_acks_late=True` + `worker_prefetch_multiplier=1` is the safe-
interruption pairing this project needs: a task is only acknowledged
(removed from the queue) after it finishes, so a worker that's killed
mid-task (deploy, OOM, crash) leaves the task requeued for another worker
rather than silently losing it; prefetch=1 stops one worker from hoarding
several minutes-long ML tasks in its local buffer while sibling workers on
the same queue sit idle.

Redis-broker redelivery, and why `visibility_timeout` is set explicitly:
unlike an AMQP broker (which detects a dropped TCP connection and requeues
immediately), Kombu's Redis transport decides a delivered-but-unacked
message is eligible for redelivery purely by *time* -
`broker_transport_options["visibility_timeout"]` seconds after delivery,
regardless of why it's still unacked. Left at Kombu's own default (3600s =
1 hour), a worker that's truly killed mid-task (SIGKILL/OOM/machine crash -
`Task.retry()` and every classified failure path in workers/tasks.py
already handle every OTHER failure mode without touching this at all) would
sit unrecovered for up to an hour before anything else could pick it back
up. Set here to twice `CELERY_TASK_TIME_LIMIT_SECONDS` - long enough that a
task which is still genuinely, legitimately running is never mistaken for
lost and redelivered out from under itself (the single biggest risk with a
short visibility_timeout - see Celery's own docs on this exact trade-off),
short enough that a genuine crash recovers in minutes instead of an hour.

`task_reject_on_worker_lost` is deliberately left at Celery's default
(`False`) rather than flipped to `True`: this configuration's actual
recovery path is the `visibility_timeout` mechanism above, which already
applies uniformly with either setting for the Redis transport - flipping it
doesn't add real coverage here, and this project would rather have one
well-understood, documented recovery mechanism than two overlapping ones.

Two real gaps this configuration alone doesn't close, both now fixed
elsewhere (updated - final hardening pass):

1. A `pipeline_runs` row left stuck at "running" forever because its
   owning worker died in a way that bypassed even the broker's own
   redelivery (e.g. the message itself was lost, not just delayed) - closed
   by `workers/reconciliation.py::reconcile_stale_pipeline_runs()`, now
   running automatically on a real Celery Beat schedule
   (`celery worker -B`, `celery_app.conf.beat_schedule` below - Beat is a
   scheduling mode of the same `celery` package this project already
   depends on, not new infrastructure) as well as remaining invocable by
   hand via `python -m voxmind.workers.reconciliation`. See docs/celery.md's
   "Reconciliation" and "Automatic scheduling" sections and
   docs/DECISIONS/0017.
2. Nothing capped how many times a message that keeps crashing whatever
   worker picks it up (a genuine "poison pill" input) could be
   redelivered - Celery's Redis transport has no such cap independent of
   `CELERY_TASK_MAX_RETRIES` (which only ever governed *caught, in-process*
   retries, never broker-level redelivery after a lost worker). Closed by
   `PipelineRun.delivery_count`, a real database-backed bound checked in
   `workers/tasks.py` before any stage executes. See docs/celery.md's
   "Poison-pill bound" section for the full design.
"""
from __future__ import annotations

from celery import Celery

from voxmind.core.config import get_settings

settings = get_settings()

celery_app = Celery("voxmind")
celery_app.conf.update(
    broker_url=settings.REDIS_URL,
    result_backend=settings.REDIS_RESULT_BACKEND_URL,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # See the module docstring's "Redis-broker redelivery" section for why
    # this is set explicitly rather than left at Kombu's 3600s default.
    broker_transport_options={"visibility_timeout": settings.CELERY_TASK_TIME_LIMIT_SECONDS * 2},
    task_default_queue=settings.CELERY_QUEUE_DEFAULT,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT_SECONDS,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT_SECONDS,
    broker_connection_retry_on_startup=True,
    # A minimal, non-sensitive result stored in Redis so Celery's own
    # apply_async()/AsyncResult machinery works; the authoritative status
    # and output always live in Postgres (`pipeline_runs`), read via
    # CeleryTaskRunner.get_status()/get_result(), independent of whether
    # Redis is reachable at query time.
    result_expires=3600,
    # Explicit, not autodiscover_tasks() - this is a plain package, not a
    # Django-style app registry, so a direct import is clearer and can't
    # silently fail to find the module.
    imports=["voxmind.workers.tasks"],
    # Automatic reconciliation scheduling (final limitations-clearance
    # pass) - see workers/reconciliation.py's module docstring for why
    # this reuses Celery Beat (a built-in scheduling mode of the same
    # `celery` package already depended on, not new infrastructure)
    # instead of leaving reconciliation as a manually/externally-invoked
    # CLI only. The interval is deliberately independent of, and shorter
    # than, CELERY_RECONCILIATION_STALE_THRESHOLD_SECONDS (the age a row
    # must reach before it's considered abandoned) - this just controls
    # how often the check itself runs, not how patient it is.
    beat_schedule={
        "reconcile-stale-pipeline-runs": {
            "task": "voxmind.reconcile_stale_pipeline_runs",
            "schedule": settings.CELERY_RECONCILIATION_INTERVAL_SECONDS,
        },
    },
)
