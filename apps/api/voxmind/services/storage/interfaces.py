"""Object storage abstraction.

Two real implementations exist: a local-filesystem backend (used by default
in dev, per the approved architecture's "local dev storage" requirement) and
an S3-compatible backend (boto3, works against MinIO locally or real S3/R2 in
production). Which one is active is a config switch (`STORAGE_BACKEND`), not
a code change — every caller depends only on this interface.

Phase 1 scope: this module and both backends are fully implemented and
tested (see tests/unit/test_storage.py). What is NOT in Phase 1 is a
product-facing "upload audio" API endpoint — that belongs to Phase 2's
speech-pipeline upload stage, and building it now without the pipeline
behind it would be an unused placeholder.
"""
from __future__ import annotations

from typing import Protocol


class StorageBackend(Protocol):
    async def upload(self, key: str, data: bytes, *, content_type: str | None = None) -> None: ...

    async def download(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None: ...

    async def exists(self, key: str) -> bool: ...
