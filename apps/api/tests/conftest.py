"""Test configuration.

Environment variables are set *before* any `voxmind` module is imported,
because `voxmind.core.config.get_settings()` is an lru_cache singleton read
by several modules at import time.

Integration tests run against a real local Postgres database
(`voxmind_test`), not a mock and not SQLite - the schema uses UUID, JSONB and
(from Phase 4) pgvector types that only a real Postgres can validate.

Each test gets its own SQLAlchemy engine (NullPool, no connection pooling)
rather than sharing the application's module-level singleton engine. This
sidesteps a well-known asyncpg/pytest-asyncio interaction where a pooled
connection created on one test's event loop is reused by a later test
running on a different loop ("Future attached to a different loop") -
giving each test a fresh engine tied to its own loop is the standard fix,
and is simpler than fighting pytest-asyncio's loop-scope configuration.
The FastAPI app's `get_db` dependency is overridden to use that same
per-test engine, so API calls and direct ORM assertions within one test see
the same data.
"""
from __future__ import annotations

import os

os.environ.setdefault("ENV", "test")
os.environ.setdefault(
    "JWT_SECRET_KEY", "test-only-secret-6f3a9c2e1b4d5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e"
)
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://voxmind:voxmind@localhost:5432/voxmind_test"
)
os.environ.setdefault("COOKIE_SECURE", "false")
os.environ.setdefault("COOKIE_SAMESITE", "lax")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("STORAGE_LOCAL_ROOT", "./.data/test-storage")
os.environ.setdefault("LLM_PROVIDER", "local_dev")
# Rate limiting is disabled by default for the whole test session, the same
# reasoning as LLM_PROVIDER above: every integration test's setup calls
# /auth/register + /auth/login, and the real "auth" category is keyed by
# client IP - which httpx's ASGITransport reports as the same constant
# value for every test in the entire session. Left enabled, the very first
# handful of tests would exhaust that one shared budget and every
# subsequent test's login would fail with an unrelated 429. The dedicated
# rate-limit tests (tests/unit/test_rate_limit.py) explicitly re-enable it
# and use their own isolated categories/identities.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")

import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from voxmind.core.config import get_settings  # noqa: E402
from voxmind.db.session import get_db  # noqa: E402
from voxmind.main import app  # noqa: E402

TABLES = [
    "turns",
    "audio_assets",
    "refresh_tokens",
    "conversation_sessions",
    "model_versions",
    "evaluation_runs",
    "pipeline_runs",
    "users",
]


@pytest_asyncio.fixture
async def db_session():
    settings = get_settings()
    test_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    session_factory = async_sessionmaker(bind=test_engine, expire_on_commit=False, autoflush=False)

    async with test_engine.begin() as conn:
        await conn.execute(text(f"TRUNCATE TABLE {', '.join(TABLES)} RESTART IDENTITY CASCADE"))

    async def _override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    async with session_factory() as session:
        yield session

    app.dependency_overrides.pop(get_db, None)
    await test_engine.dispose()


@pytest_asyncio.fixture
async def client(db_session):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
