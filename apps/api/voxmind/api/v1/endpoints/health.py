from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings, get_settings
from voxmind.db.session import get_db

router = APIRouter(tags=["health"])
logger = structlog.get_logger(__name__)


def _check_redis(settings: Settings) -> str:
    """Reachability of the Celery broker itself - distinct from whether any
    worker is actually consuming from it (see `_check_celery_workers`)."""
    try:
        import redis

        client = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        client.close()
        return "ok"
    except Exception as exc:  # noqa: BLE001 - any connectivity/config failure is "unreachable", never fatal to the health endpoint
        logger.warning("health_check_redis_unreachable", error=str(exc))
        return "unreachable"


def _check_celery_workers(settings: Settings) -> str:
    """Deliberately stronger than "the broker is up": this pings live
    workers over Celery's control channel, so a broker that's reachable but
    has *no worker consuming from it* reports "no_workers" rather than a
    misleading "ok". Reports "not_configured" when this deployment isn't
    using Celery at all (TASK_RUNNER=in_process), which is a valid state,
    not a failure."""
    if settings.TASK_RUNNER != "celery":
        return "not_configured"
    try:
        from voxmind.workers.celery_app import celery_app

        replies = celery_app.control.ping(timeout=2.0)
        if not replies:
            return "no_workers"
        return "ok"
    except Exception as exc:  # noqa: BLE001
        logger.warning("health_check_celery_unreachable", error=str(exc))
        return "unreachable"


@router.get("/health")
async def health(session: AsyncSession = Depends(get_db)) -> dict:
    """Liveness of this API process plus real reachability of each
    dependency. `status` stays "ok" as long as the API process itself is
    serving - a dependency being down is reported per-dependency rather
    than collapsing the whole response, so an operator can tell *which*
    piece is broken. See `/health/ready` for the readiness (is this
    deployment actually able to do work) variant."""
    settings = get_settings()
    try:
        await session.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:
        logger.warning("health_check_db_unreachable", error=str(exc))
        db_status = "unreachable"

    return {
        "status": "ok",
        "database": db_status,
        "redis": _check_redis(settings),
        "celery_workers": _check_celery_workers(settings),
    }


@router.get("/health/ready")
async def readiness(session: AsyncSession = Depends(get_db)) -> dict:
    """Readiness: is this deployment genuinely able to serve real work right
    now? Unlike `/health` (which always returns 200 as long as the process
    is alive), `ready` is false when a dependency this deployment actually
    depends on is unavailable - Postgres always, plus Redis *and* a live
    Celery worker when TASK_RUNNER=celery. A deployment running
    TASK_RUNNER=in_process is ready without either, because it genuinely
    doesn't need them."""
    settings = get_settings()
    try:
        await session.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:
        logger.warning("readiness_check_db_unreachable", error=str(exc))
        db_status = "unreachable"

    redis_status = _check_redis(settings) if settings.TASK_RUNNER == "celery" else "not_configured"
    worker_status = _check_celery_workers(settings)

    checks = {"database": db_status, "redis": redis_status, "celery_workers": worker_status}
    ready = db_status == "ok" and all(
        status in ("ok", "not_configured") for status in (redis_status, worker_status)
    )
    return {"ready": ready, "checks": checks}
