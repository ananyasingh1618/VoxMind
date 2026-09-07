"""Redis-backed, distributed application rate limiting.

This protects VoxMind's own API from abuse and runaway expensive requests -
it is never applied to Groq/Hugging Face calls themselves, and never to
internal Celery task execution (a worker consuming from its own queue is
not a "request" in this sense at all).

Algorithm: a fixed-window counter, the simplest genuinely distributed Redis
rate-limiting pattern (a Lua script does `INCR` then, only on the very
first increment of a window, `EXPIRE` - atomic, one round trip, easy to
reason about and explain). The one honest trade-off, disclosed rather than
hidden: a client can get up to ~2x the configured limit's worth of requests
across a single window *boundary* (e.g. a burst just before a window ends,
followed immediately by a fresh burst once it resets) - a sliding-window
log or a token bucket avoids that at the cost of materially more
complexity. For an abuse guard (not a precise billing meter), that
trade-off is the right one; see docs/rate_limiting.md.

Identity: an authenticated request is keyed by user id (`user:<uuid>`) -
strictly more meaningful than any IP, since one user can legitimately sit
behind a shared/rotating IP and one abusive IP can host many accounts. An
unauthenticated request (only the auth-category endpoints reach this
module without a user) is keyed by client IP (`ip:<address>`). See
`_client_ip()` for exactly why `request.client.host`, not a
client-supplied `X-Forwarded-For` header, is trusted by default.

Redis client: `redis.asyncio` (redis-py's own async client), never the
synchronous `redis.Redis` client `health.py` uses for its one-shot liveness
ping - a rate-limit check runs on every gated request and must never block
the event loop.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any, cast

import redis.asyncio as redis_asyncio
import structlog
from fastapi import Depends, Request

from voxmind.api.deps import get_current_user
from voxmind.core.config import Settings, get_settings
from voxmind.core.exceptions import RateLimitExceededError
from voxmind.models.user import User

logger = structlog.get_logger(__name__)

# Atomic: a second request arriving between INCR and EXPIRE could otherwise
# either race past the limit uncounted, or (if the process died between the
# two calls) leave a counter with no TTL that never resets. One Lua script
# makes both calls a single atomic Redis operation.
_INCR_AND_EXPIRE_LUA = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""


@lru_cache
def get_rate_limit_redis() -> redis_asyncio.Redis:
    """Process-wide singleton connection pool, the same lifecycle pattern
    `api/deps.py::get_storage_backend()` already uses for a resource that
    must be created once and reused - not per-request. Tests override this
    via `app.dependency_overrides` exactly like every other singleton
    dependency in this codebase, never by swapping in an in-memory fake
    that would stop testing real Redis behavior."""
    settings = get_settings()
    return redis_asyncio.Redis.from_url(
        settings.REDIS_RATE_LIMIT_URL,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )


def _client_ip(request: Request, settings: Settings) -> str:
    if settings.RATE_LIMIT_TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # The first hop is the original client in the standard
            # left-to-right convention - only meaningful once a real,
            # controlling proxy (not the client) is the one setting it.
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _enforce(*, category: str, identity: str, limit: int, settings: Settings) -> None:
    if not settings.RATE_LIMIT_ENABLED:
        return

    key = f"ratelimit:{category}:{identity}"
    redis_client = get_rate_limit_redis()
    try:
        # redis-py's EVAL requires every KEYS/ARGV entry as a string - the
        # window is passed as one even though it's an int in Settings.
        # `Awaitable[Any]` cast: this client is genuinely the async one
        # (constructed via `redis.asyncio.Redis.from_url` above), but
        # redis-py's stubs type `eval()` as a sync/async union its type
        # checker can't narrow from this call site alone.
        current = int(
            await cast(
                "Awaitable[Any]",
                redis_client.eval(_INCR_AND_EXPIRE_LUA, 1, key, str(settings.RATE_LIMIT_WINDOW_SECONDS)),
            )
        )
    except Exception as exc:  # noqa: BLE001 - any connectivity/protocol failure from redis-py
        # Never silent either way - a Redis outage is always logged, whether
        # this deployment then chooses to fail open or fail closed.
        logger.warning(
            "rate_limit_backend_unreachable",
            category=category,
            fail_open=settings.RATE_LIMIT_FAIL_OPEN,
            error_type=type(exc).__name__,
        )
        if settings.RATE_LIMIT_FAIL_OPEN:
            return
        raise RateLimitExceededError(
            "Rate limiting is temporarily unavailable and this deployment is configured to "
            "fail closed rather than run unprotected.",
            retry_after_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        ) from exc

    if current <= limit:
        return

    ttl = await redis_client.ttl(key)
    retry_after = ttl if ttl and ttl > 0 else settings.RATE_LIMIT_WINDOW_SECONDS
    logger.info("rate_limit_exceeded", category=category, identity=identity, limit=limit)
    raise RateLimitExceededError(
        f"Too many requests for '{category}'. Please slow down and try again shortly.",
        retry_after_seconds=retry_after,
    )


def rate_limit_by_ip(category: str, limit_setting: str) -> Callable[..., Awaitable[None]]:
    """For endpoints reached before any authentication exists - currently
    only register/login/refresh. `limit_setting` is the `Settings` field
    name to read the real limit from (e.g. "RATE_LIMIT_AUTH_PER_WINDOW"),
    never a number hardcoded at the call site."""

    async def _dependency(request: Request, settings: Settings = Depends(get_settings)) -> None:
        limit = getattr(settings, limit_setting)
        await _enforce(
            category=category, identity=f"ip:{_client_ip(request, settings)}", limit=limit, settings=settings
        )

    return _dependency


def rate_limit_by_user(category: str, limit_setting: str) -> Callable[..., Awaitable[None]]:
    """For already-authenticated endpoints. Depends on the same
    `get_current_user` the route itself already depends on - FastAPI
    resolves a given dependency callable at most once per request, so this
    never re-decodes the JWT or hits the database a second time."""

    async def _dependency(
        current_user: User = Depends(get_current_user), settings: Settings = Depends(get_settings)
    ) -> None:
        limit = getattr(settings, limit_setting)
        await _enforce(category=category, identity=f"user:{current_user.id}", limit=limit, settings=settings)

    return _dependency
