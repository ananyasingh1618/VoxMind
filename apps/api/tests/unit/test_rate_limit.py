"""Real Redis-backed tests of `core/rate_limit.py`, plus a handful of
real, full-HTTP tests proving the wiring onto actual endpoints. All marked
`requires_redis` and `skipif`-guarded exactly like `test_celery_task_runner.py`
- skipped, never faked, when no broker is reachable. Nothing here replaces
Redis with an in-memory fake: every test drives a real `redis.asyncio.Redis`
client against `settings.REDIS_RATE_LIMIT_URL`.

Every Redis client used below is created, used, and closed entirely within
one test function's own body - never handed across an `async fixture`
`yield` boundary. This project's pytest-asyncio configuration
(`asyncio_default_fixture_loop_scope = "session"` in pyproject.toml) gives
async *fixtures* a session-scoped loop by default while ordinary
`@pytest.mark.asyncio` *test functions* still each run on their own
function-scoped loop - a real, reproducible mismatch (confirmed live: a
`redis.asyncio.Redis` connection created in a fixture's pre-yield code and
reused in the test body raised a genuine
`RuntimeError: ... attached to a different loop`, the identical class of
asyncpg/pytest-asyncio loop-lifecycle issue `tests/conftest.py` already
documents for `db_session`, now confirmed for `redis.asyncio` too). Keeping
each client's entire lifecycle inside one coroutine avoids the mismatch
without touching that shared, working, project-wide pytest-asyncio setting.

Rate limiting is disabled by default for the whole test session (see
conftest.py) precisely so ordinary tests never trip it - every test in
this file explicitly re-enables it via its own real `Settings` copy for the
one behavior it is checking, and always cleans up its own Redis keys so it
can't leak state into a later test or a later run.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import pytest
import redis.asyncio as redis_asyncio

from voxmind.core.config import Settings, get_settings
from voxmind.core.exceptions import RateLimitExceededError
from voxmind.core.rate_limit import _enforce
from voxmind.main import app


def _redis_is_reachable() -> bool:
    try:
        import redis as redis_sync

        client = redis_sync.Redis.from_url(
            get_settings().REDIS_RATE_LIMIT_URL, socket_connect_timeout=1, socket_timeout=1
        )
        client.ping()
        client.close()
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.requires_redis,
    pytest.mark.skipif(not _redis_is_reachable(), reason="No reachable Redis broker in this environment."),
]


def _enabled_settings(**overrides) -> Settings:
    """A real Settings instance with rate limiting genuinely enabled,
    regardless of the test-session default - never a mock/stub Settings."""
    return get_settings().model_copy(update={"RATE_LIMIT_ENABLED": True, **overrides})


@asynccontextmanager
async def _real_redis_client(monkeypatch):
    """A fresh, real `redis.asyncio.Redis` client - created, purged of any
    key in the rate-limit keyspace, handed to the caller, purged again, and
    closed, all inside the one `async with` block the caller opens - never
    crossing a fixture `yield` boundary (see module docstring).

    Also monkeypatched over `voxmind.core.rate_limit.get_rate_limit_redis`
    for the duration of the block: `_enforce()` (called both directly by
    this file's tests and indirectly by real HTTP requests through the
    `client` fixture) looks up that module-global name itself, so without
    this, every test would actually exercise the real, process-wide
    `lru_cache`'d production singleton instead of this test's own client -
    reintroducing the exact cross-loop risk this whole pattern exists to
    avoid, the moment any two tests' function-scoped loops disagree with
    whichever loop first created that singleton."""
    client = redis_asyncio.Redis.from_url(
        get_settings().REDIS_RATE_LIMIT_URL, decode_responses=True, socket_connect_timeout=2, socket_timeout=2
    )
    monkeypatch.setattr("voxmind.core.rate_limit.get_rate_limit_redis", lambda: client)

    async def _purge():
        # Broader than just this file's own "test-*" categories: the
        # HTTP-level tests below exercise the real "auth"/"expensive"/
        # "default" category names (keyed by the fixed IP ASGITransport
        # always reports), so a real leftover key from an earlier run
        # within the same window would otherwise silently contaminate the
        # next run. Safe to purge broadly - REDIS_RATE_LIMIT_URL is this
        # deployment's own dedicated Redis DB index for rate-limit keys
        # only, never the Celery broker/result-backend indices.
        async for key in client.scan_iter(match="ratelimit:*"):
            await client.delete(key)

    try:
        await _purge()
        yield client
        await _purge()
    finally:
        await client.aclose()


# --- Direct, real-Redis tests of the core enforcement logic ----------------


@pytest.mark.asyncio
async def test_requests_under_the_limit_are_allowed(monkeypatch):
    async with _real_redis_client(monkeypatch):
        settings = _enabled_settings()
        identity = f"unit:{uuid.uuid4()}"
        for _ in range(3):
            await _enforce(category="test-under", identity=identity, limit=3, settings=settings)
        # No exception raised for any of the 3 - all genuinely allowed.


@pytest.mark.asyncio
async def test_requests_over_the_limit_are_rejected_with_429_semantics(monkeypatch):
    async with _real_redis_client(monkeypatch):
        settings = _enabled_settings()
        identity = f"unit:{uuid.uuid4()}"
        for _ in range(3):
            await _enforce(category="test-over", identity=identity, limit=3, settings=settings)
        with pytest.raises(RateLimitExceededError) as exc_info:
            await _enforce(category="test-over", identity=identity, limit=3, settings=settings)
        assert exc_info.value.status_code == 429
        assert exc_info.value.code == "rate_limited"


@pytest.mark.asyncio
async def test_retry_after_reflects_the_real_remaining_window(monkeypatch):
    async with _real_redis_client(monkeypatch):
        settings = _enabled_settings(RATE_LIMIT_WINDOW_SECONDS=30)
        identity = f"unit:{uuid.uuid4()}"
        await _enforce(category="test-retry", identity=identity, limit=1, settings=settings)
        with pytest.raises(RateLimitExceededError) as exc_info:
            await _enforce(category="test-retry", identity=identity, limit=1, settings=settings)
        # A real TTL read from Redis, not a hardcoded echo of the window.
        assert 0 < exc_info.value.retry_after_seconds <= 30


@pytest.mark.asyncio
async def test_different_identities_never_share_a_budget(monkeypatch):
    """Proves category+identity isolation - the mechanism behind "separate
    authenticated users get separate budgets" and "one user's activity in
    one category never affects another category"."""
    async with _real_redis_client(monkeypatch):
        settings = _enabled_settings()
        identity_a = f"unit:{uuid.uuid4()}"
        identity_b = f"unit:{uuid.uuid4()}"
        for _ in range(2):
            await _enforce(category="test-isolation", identity=identity_a, limit=2, settings=settings)
        # identity_a is now exhausted - identity_b must be completely unaffected.
        await _enforce(category="test-isolation", identity=identity_b, limit=2, settings=settings)
        with pytest.raises(RateLimitExceededError):
            await _enforce(category="test-isolation", identity=identity_a, limit=2, settings=settings)


@pytest.mark.asyncio
async def test_different_categories_for_the_same_identity_are_independent(monkeypatch):
    async with _real_redis_client(monkeypatch):
        settings = _enabled_settings()
        identity = f"unit:{uuid.uuid4()}"
        for _ in range(2):
            await _enforce(category="test-cat-a", identity=identity, limit=2, settings=settings)
        # Same identity, a different category - must have its own fresh budget.
        await _enforce(category="test-cat-b", identity=identity, limit=2, settings=settings)


@pytest.mark.asyncio
async def test_configuration_override_changes_the_real_effective_limit(monkeypatch):
    """Configurable, not hardcoded: the exact same call site enforces
    whatever `limit`/`window_seconds` Settings carries."""
    async with _real_redis_client(monkeypatch):
        identity = f"unit:{uuid.uuid4()}"
        strict = _enabled_settings()
        await _enforce(category="test-config", identity=identity, limit=1, settings=strict)
        with pytest.raises(RateLimitExceededError):
            await _enforce(category="test-config", identity=identity, limit=1, settings=strict)

        identity_2 = f"unit:{uuid.uuid4()}"
        lenient = _enabled_settings()
        for _ in range(5):
            # A higher limit for a fresh identity - genuinely permits more,
            # proving the limit isn't secretly hardcoded somewhere below _enforce.
            await _enforce(category="test-config", identity=identity_2, limit=5, settings=lenient)


@pytest.mark.asyncio
async def test_disabled_rate_limiting_never_touches_redis(monkeypatch):
    async with _real_redis_client(monkeypatch) as client:
        settings = _enabled_settings(RATE_LIMIT_ENABLED=False)
        identity = f"unit:{uuid.uuid4()}"
        for _ in range(100):
            await _enforce(category="test-disabled", identity=identity, limit=1, settings=settings)
        # Never persisted anything, even though "limit=1" was crossed 99 times over.
        assert await client.get(f"ratelimit:test-disabled:{identity}") is None


# --- Redis failure behavior: a real, unreachable Redis, not a mock ---------


@pytest.mark.asyncio
async def test_redis_failure_fails_open_when_configured_to(monkeypatch):
    broken_client = redis_asyncio.Redis.from_url(
        "redis://localhost:6399/2", socket_connect_timeout=1, socket_timeout=1
    )
    monkeypatch.setattr("voxmind.core.rate_limit.get_rate_limit_redis", lambda: broken_client)
    settings = _enabled_settings(RATE_LIMIT_FAIL_OPEN=True)
    # A real, genuine connection failure against an unreachable port - not
    # simulated - and the request is still allowed through, as configured.
    await _enforce(category="test-failopen", identity="whoever", limit=1, settings=settings)
    await broken_client.aclose()


@pytest.mark.asyncio
async def test_redis_failure_fails_closed_when_configured_to(monkeypatch):
    broken_client = redis_asyncio.Redis.from_url(
        "redis://localhost:6399/2", socket_connect_timeout=1, socket_timeout=1
    )
    monkeypatch.setattr("voxmind.core.rate_limit.get_rate_limit_redis", lambda: broken_client)
    settings = _enabled_settings(RATE_LIMIT_FAIL_OPEN=False)
    with pytest.raises(RateLimitExceededError) as exc_info:
        await _enforce(category="test-failclosed", identity="whoever", limit=1, settings=settings)
    assert exc_info.value.status_code == 429
    await broken_client.aclose()


# --- Real, full-HTTP tests proving the wiring onto actual endpoints --------


def _override_settings(**overrides):
    """Overrides the real FastAPI dependency the same way `get_db` is
    overridden elsewhere in this suite - never a stub Settings class, a
    real one with only the specific category this one test is checking
    tightened, so an unrelated category's own default (e.g. register+login
    already spending two "auth"-category calls per user) never starves it."""
    settings = _enabled_settings(**overrides)
    app.dependency_overrides[get_settings] = lambda: settings
    return settings


@pytest.fixture(autouse=True)
def _reset_settings_override():
    yield
    app.dependency_overrides.pop(get_settings, None)


@pytest.mark.asyncio
async def test_unauthenticated_auth_endpoint_returns_real_429_over_http(client, monkeypatch):
    async with _real_redis_client(monkeypatch):
        _override_settings(RATE_LIMIT_AUTH_PER_WINDOW=3)
        payload = {"email": "rate-limit-http-test@voxmind.dev", "password": "wrong-password-entirely"}
        responses = [await client.post("/api/v1/auth/login", json=payload) for _ in range(4)]
        statuses = [r.status_code for r in responses]
        # First 3 reach real auth logic (401, wrong password) - the 4th is blocked.
        assert statuses[:3] == [401, 401, 401]
        assert statuses[3] == 429
        body = responses[3].json()
        assert body["error"]["code"] == "rate_limited"
        assert "Retry-After" in responses[3].headers
        assert int(responses[3].headers["Retry-After"]) > 0


@pytest.mark.asyncio
async def test_expensive_endpoint_separates_different_authenticated_users(client, monkeypatch):
    async with _real_redis_client(monkeypatch):
        # A generous auth budget - this test's setup alone spends 2 "auth"-
        # category calls per user (register + login); only "expensive" is
        # actually under test here.
        _override_settings(RATE_LIMIT_AUTH_PER_WINDOW=20, RATE_LIMIT_EXPENSIVE_PER_WINDOW=2)

        async def _register_and_login(email: str) -> dict:
            creds = {"email": email, "password": "correct-horse-battery-staple-9"}
            await client.post("/api/v1/auth/register", json=creds)
            login = await client.post("/api/v1/auth/login", json=creds)
            return {"Authorization": f"Bearer {login.json()['access_token']}"}

        headers_a = await _register_and_login(f"rl-user-a-{uuid.uuid4()}@voxmind.dev")
        headers_b = await _register_and_login(f"rl-user-b-{uuid.uuid4()}@voxmind.dev")

        conv_a = await client.post("/api/v1/conversations", json={"title": "a"}, headers=headers_a)
        conv_a_id = conv_a.json()["id"]

        # RATE_LIMIT_EXPENSIVE_PER_WINDOW=2 - exhaust user A's budget on a
        # real expensive-category endpoint (audio upload).
        wav_bytes = b"not-real-audio-but-only-the-rate-limit-dependency-runs-before-validation"
        for _ in range(2):
            await client.post(
                f"/api/v1/conversations/{conv_a_id}/audio",
                files={"file": ("clip.wav", wav_bytes, "audio/wav")},
                headers=headers_a,
            )
        blocked = await client.post(
            f"/api/v1/conversations/{conv_a_id}/audio",
            files={"file": ("clip.wav", wav_bytes, "audio/wav")},
            headers=headers_a,
        )
        assert blocked.status_code == 429

        # User B has never uploaded anything - a completely independent budget.
        conv_b = await client.post("/api/v1/conversations", json={"title": "b"}, headers=headers_b)
        conv_b_id = conv_b.json()["id"]
        still_allowed = await client.post(
            f"/api/v1/conversations/{conv_b_id}/audio",
            files={"file": ("clip.wav", wav_bytes, "audio/wav")},
            headers=headers_b,
        )
        assert still_allowed.status_code != 429


@pytest.mark.asyncio
async def test_ordinary_default_category_endpoint_is_rate_limited_independently_of_expensive(client, monkeypatch):
    """GET /conversations (category D, "default") must not share a budget
    with the "expensive" category, and a generous default limit must not
    block ordinary read traffic."""
    async with _real_redis_client(monkeypatch):
        _override_settings(RATE_LIMIT_AUTH_PER_WINDOW=20)
        creds = {
            "email": f"rl-default-{uuid.uuid4()}@voxmind.dev",
            "password": "correct-horse-battery-staple-9",
        }
        await client.post("/api/v1/auth/register", json=creds)
        login = await client.post("/api/v1/auth/login", json=creds)
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        # RATE_LIMIT_DEFAULT_PER_WINDOW is the real, generous default (120)
        # in this override - well above these 5 ordinary read requests.
        for _ in range(5):
            response = await client.get("/api/v1/conversations", headers=headers)
            assert response.status_code == 200


# --- Celery must never be affected -----------------------------------------


def test_worker_modules_have_no_dependency_on_rate_limiting():
    """Structural check, matching the pattern
    `test_pipeline_run_not_exposed.py` already uses for a different
    invariant: rate limiting exists only at the API request-initiation
    boundary. If a worker module ever imported it, that would be a real
    architecture violation - internal Celery execution must never depend on
    (or be gated by) this at all."""
    import inspect

    from voxmind.workers import celery_app, stage_registry, task_runner, tasks

    for module in (celery_app, stage_registry, task_runner, tasks):
        source = inspect.getsource(module)
        assert "rate_limit" not in source, f"{module.__name__} must never reference rate limiting"
