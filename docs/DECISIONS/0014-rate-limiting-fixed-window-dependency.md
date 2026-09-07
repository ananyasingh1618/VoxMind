# ADR 0014: rate limiting as a FastAPI dependency, not middleware; a fixed-window counter, not a sliding window or token bucket

## The choice: dependency, not middleware

`RequestContextMiddleware` (Phase 1, hardened in Phase 5) is deliberately plain ASGI middleware rather than a `starlette.middleware.base.BaseHTTPMiddleware` subclass, because that base class has a documented Starlette bug that can hang a streaming response's disconnect handling - found for real while building the Phase 5 voice loop (see [DECISIONS/0009](0009-voice-loop-cancellation.md)). Rate limiting has the exact same streaming exposure: `/voice-turns` returns a `StreamingResponse`, and the requirement is to gate *initiation* only, never individual stream events.

Rather than writing a second piece of careful ASGI middleware and re-deriving the same streaming-safety guarantees `RequestContextMiddleware` already had to earn the hard way, rate limiting is implemented as an ordinary FastAPI dependency (`core/rate_limit.py::rate_limit_by_ip` / `rate_limit_by_user`), attached via `dependencies=[Depends(...)]` on specific routes and via `include_router(..., dependencies=[...])` for the router-wide default category. FastAPI resolves every dependency *before* the route handler body runs - including before a `StreamingResponse`'s async generator is ever constructed - so this is structurally incapable of ever touching an in-flight stream, with no bespoke ASGI code required at all.

## The choice: fixed-window counter, not sliding window or token bucket

A sliding-window log or a token bucket both avoid the fixed-window counter's one real weakness (up to ~2x the configured budget across a single window boundary), at the cost of materially more Redis round trips and more state to reason about. For an abuse guard - not a precise billing meter - that trade-off is the wrong one to pay for. The fixed-window counter is one Redis Lua script (`INCR` then, only on the very first increment, `EXPIRE`, made atomic in a single round trip), it is trivial to explain, and its one weakness is disclosed plainly in [docs/rate_limiting.md](../rate_limiting.md) rather than hidden.

## Identity: user id where available, IP only where it must be

An authenticated request is keyed by user id, not IP - a shared or rotating IP address behind one legitimate user, or one abusive IP hosting many accounts, would otherwise either wrongly conflate different real users or wrongly under-protect against one bad one. Only the three endpoints that run before any user exists (register/login/refresh) fall back to IP.

`X-Forwarded-For` is not trusted by default. This project's own compose files publish the API container's port directly - no reverse proxy sits in front of it in any deployment shipped here - so trusting a client-supplied header would let any caller spoof their own rate-limit identity outright. `RATE_LIMIT_TRUST_PROXY_HEADERS` exists as a deliberate, off-by-default escape hatch for a real deployment that adds a proxy which sets the header itself.

## A real, reproducible pytest-asyncio/redis.asyncio bug found while testing this

This project's `pyproject.toml` sets `asyncio_default_fixture_loop_scope = "session"`, giving async *fixtures* a session-scoped event loop by default - but ordinary `@pytest.mark.asyncio` *test functions* still each run on their own function-scoped loop. A `redis.asyncio.Redis` connection created in an async fixture's pre-`yield` code and then reused inside the test body genuinely raised `RuntimeError: ... attached to a different loop` - the identical class of asyncpg/pytest-asyncio loop-lifecycle issue `tests/conftest.py` already documents for `db_session`, now reproduced for `redis.asyncio` as well, and previously never exercised because no earlier test in this project drove a real async Redis client through a `yield`-based fixture at all. `tests/unit/test_rate_limit.py` works around it the same way `db_session` already does: every Redis client used by a test is created, used, and closed entirely within that one test function's own body, via a small `_real_redis_client()` async context manager rather than a fixture that spans the `yield` boundary - never by changing the project-wide pytest-asyncio setting, which the rest of the suite already depends on working the way it does.

## Verification

Live, against a real Redis instance, not simulated: real under/over-limit counting, a real `Retry-After` value read via Redis `TTL`, real category/identity isolation, both real Redis-failure behaviors (fail-open and fail-closed) against a genuinely unreachable port, and three full-HTTP tests (a real `429` with the real error envelope on `/auth/login`, two authenticated users never sharing an "expensive"-category budget on real audio upload, and ordinary `/conversations` reads never hitting the stricter category). A structural test confirms zero Celery worker module ever references rate limiting at all.
