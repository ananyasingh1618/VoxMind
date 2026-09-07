# Rate limiting

Protects VoxMind's own API from abuse and runaway expensive requests. This is **not** a rate limiter for Groq or Hugging Face - those services enforce their own limits independently - and it is **never** applied to internal Celery task execution: a worker consuming a message from its own queue is not a "request" in the sense this document covers at all (see [Celery](#celery) below).

## Algorithm

A **fixed-window counter**, implemented with one Redis Lua script per check:

```lua
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
```

`INCR` and the first-time `EXPIRE` happen as one atomic Redis operation - a second request racing the first can never slip past uncounted, and a process crash between the two calls can never leave a counter with no expiry that never resets. This is the simplest genuinely distributed rate-limiting pattern available on Redis, and the one honest trade-off is disclosed rather than hidden: a client can receive up to roughly 2x its configured budget across a single window *boundary* (a burst just before a window ends, followed immediately by a fresh burst once it resets). A sliding-window log or a token bucket avoids that at real additional complexity. For an abuse guard - not a precise billing meter - the fixed-window counter is the right trade-off.

## Endpoint categories

| Category | Endpoints | Identity | Default limit |
|---|---|---|---|
| **A - Auth** | `POST /auth/register`, `/auth/login`, `/auth/refresh` | client IP | `RATE_LIMIT_AUTH_PER_WINDOW` = 10 |
| **B - Expensive** | audio upload/process, `/voice-turns`, emotion processing, `/ask` | authenticated user | `RATE_LIMIT_EXPENSIVE_PER_WINDOW` = 15 |
| **C - Knowledge** | document upload/process | authenticated user | `RATE_LIMIT_KNOWLEDGE_PER_WINDOW` = 20 |
| **D - Default** | every other authenticated endpoint (not `/health`) | authenticated user | `RATE_LIMIT_DEFAULT_PER_WINDOW` = 120 |

All four categories share one window size, `RATE_LIMIT_WINDOW_SECONDS` (default 60s) - one setting to reason about rather than a separate one per category, deliberately kept simple.

Categories are independent: an endpoint carrying both a stricter category (B/C) and the router-wide default (D) is checked against both, and the stricter one always binds first in practice - this is intentional, harmless layering, not a bug. `/health` and `/health/ready` are never rate-limited at all - an orchestrator's healthcheck polling them frequently is exactly the traffic they exist to always answer.

## Identity

An authenticated request is keyed by **user id** (`ratelimit:<category>:user:<uuid>`) - more meaningful than an IP, since one legitimate user can sit behind a shared or rotating address and one abusive address can host many accounts. The three unauthenticated auth-category endpoints are keyed by **client IP** (`ratelimit:auth:ip:<address>`), since no user exists yet at that point.

**Proxy headers are not trusted by default.** No deployment this project ships puts a reverse proxy in front of the API (`docker-compose.full.yml` publishes the API container's port directly), so trusting a client-supplied `X-Forwarded-For` header would let any caller trivially spoof their own rate-limit identity. `RATE_LIMIT_TRUST_PROXY_HEADERS` exists for a real deployment that adds a proxy which sets that header itself - only flip it once one is guaranteed to be there.

## Configuration

All of it lives in `Settings` (`core/config.py`), never hardcoded at a call site:

| Setting | Default | Meaning |
|---|---|---|
| `RATE_LIMIT_ENABLED` | `true` | master on/off switch |
| `REDIS_RATE_LIMIT_URL` | `redis://localhost:6379/2` | a third logical Redis DB index, dedicated to rate-limit counters - the same broker (`/0`)/result-backend (`/1`) separation convention `workers/celery_app.py` already established, extended to a third, unrelated traffic type. Not a new database or new infrastructure. |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | shared window size across all categories |
| `RATE_LIMIT_FAIL_OPEN` | `true` | see [Redis failure behavior](#redis-failure-behavior) |
| `RATE_LIMIT_TRUST_PROXY_HEADERS` | `false` | see [Identity](#identity) |
| `RATE_LIMIT_AUTH_PER_WINDOW` | `10` | category A |
| `RATE_LIMIT_EXPENSIVE_PER_WINDOW` | `15` | category B |
| `RATE_LIMIT_KNOWLEDGE_PER_WINDOW` | `20` | category C |
| `RATE_LIMIT_DEFAULT_PER_WINDOW` | `120` | category D |

## Response behavior

An over-limit request receives a real `429 Too Many Requests` with the project's standard error envelope:

```json
{"data": null, "error": {"code": "rate_limited", "message": "Too many requests for 'expensive'. Please slow down and try again shortly.", "detail": null}, "request_id": "..."}
```

plus a real `Retry-After` header - read from Redis's own `TTL` on the counter key, never a hardcoded echo of the configured window. No internal detail (the real count, the real limit, which Redis key) is ever included in the response body.

## Redis failure behavior

Never silent either way - a Redis outage is always logged (`rate_limit_backend_unreachable`, with the exception type only, never its full message, in case a future Redis client version's error text ever embedded connection credentials). What happens next is explicit and configurable via `RATE_LIMIT_FAIL_OPEN`:

- **Fail open (default, `true`)**: the request is allowed through. The right default for an abuse guard, not a security boundary like authentication - a transient Redis blip should degrade to "temporarily unprotected," not "API down."
- **Fail closed (`false`)**: the request is rejected with `429` and an honest message explaining rate limiting itself is unavailable. For a deployment that would rather reject traffic outright than run even briefly unprotected.

Verified live: a real, unreachable Redis URL genuinely produces both behaviors as configured, not simulated.

## Streaming / SSE

The dependency runs once, before the route handler starts - including before a `StreamingResponse`'s async generator begins. This means `/voice-turns` is rate-limited at *initiation* only: once a stream has started, nothing about this mechanism observes or interferes with its individual events. This also matters architecturally: this project deliberately does not implement `RequestContextMiddleware` (or anything else touching the request/response cycle) as `starlette.middleware.base.BaseHTTPMiddleware`, because that base class has a documented bug that can hang a streaming response's disconnect handling (see [DECISIONS/0009](DECISIONS/0009-voice-loop-cancellation.md)) - rate limiting is implemented as a FastAPI dependency, not middleware, specifically so it can never reintroduce that class of problem.

## Celery

Rate limiting exists only at the API request-initiation boundary - `workers/celery_app.py`, `workers/tasks.py`, `workers/stage_registry.py`, and `workers/task_runner.py` have zero dependency on it (enforced by a structural test, `test_worker_modules_have_no_dependency_on_rate_limiting`). A Celery worker consuming its own queue is never gated by this at all.

## Scaling behavior

Because the counters live in Redis rather than in-process memory, the limits are correctly enforced across any number of API processes/replicas sharing the same `REDIS_RATE_LIMIT_URL` - a request handled by replica 1 and the next one handled by replica 2, for the same user within the same window, still share one real budget.

## Testing

`tests/unit/test_rate_limit.py` - real Redis (marked `requires_redis`, skipped honestly without a reachable broker, never faked): under/over-limit behavior, real `Retry-After` values read from Redis, identity/category isolation, configuration overrides, disabled-mode never touching Redis, both real Redis-failure behaviors (a genuinely unreachable port, not a mock), real HTTP-level `429`s with the real error envelope and header, separate authenticated users never sharing a budget, and the structural Celery-independence check above.
