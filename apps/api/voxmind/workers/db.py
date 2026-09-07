"""Worker-local database engine (Phase 9) - deliberately separate from
`voxmind.db.session`'s pooled engine, which the API process uses across
many requests inside one long-lived event loop.

A Celery task under the prefork pool runs inside a long-lived child
process but each task invocation calls `asyncio.run(...)` (see
`workers/tasks.py`), which creates and tears down its own event loop every
time. asyncpg connections are bound to the event loop that created them, so
reusing a pooled connection across two different `asyncio.run()` calls in
the same worker process raises a real "attached to a different loop" error
- this is the same class of asyncpg/event-loop interaction already
documented in `apps/api/tests/conftest.py`. `NullPool` sidesteps it
entirely: every checkout opens a fresh physical connection and every
checkin closes it, so nothing is ever held across event loops.

The engine itself is created lazily (on first use inside a running event
loop), not at module import time - constructing it eagerly at import time
proved to interact badly with pytest-asyncio's own event-loop lifecycle in
tests that merely import this module without using it (see the Phase 9
test suite), and lazy construction is also simply more correct for the
worker's real usage pattern: there is no event loop at all until
`asyncio.run(...)` starts one.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from voxmind.core.config import get_settings

_worker_engine: AsyncEngine | None = None


def _get_worker_engine() -> AsyncEngine:
    global _worker_engine
    if _worker_engine is None:
        _worker_engine = create_async_engine(get_settings().DATABASE_URL, poolclass=NullPool)
    return _worker_engine


@asynccontextmanager
async def worker_session_scope() -> AsyncGenerator[AsyncSession, None]:
    session_factory = async_sessionmaker(bind=_get_worker_engine(), expire_on_commit=False, autoflush=False)
    async with session_factory() as session:
        yield session
