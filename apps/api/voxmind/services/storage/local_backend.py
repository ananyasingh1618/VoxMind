from __future__ import annotations

import asyncio
from pathlib import Path


class LocalFilesystemStorage:
    """Real filesystem storage for local development. Not a mock — actual
    files are written to and read from disk under `root`."""

    def __init__(self, root: str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if not str(path).startswith(str(self._root.resolve())):
            raise ValueError(f"Storage key escapes storage root: {key!r}")
        return path

    async def upload(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        path = self._resolve(key)
        await asyncio.to_thread(self._write, path, data)

    def _write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def download(self, key: str) -> bytes:
        path = self._resolve(key)
        return await asyncio.to_thread(path.read_bytes)

    async def delete(self, key: str) -> None:
        path = self._resolve(key)
        await asyncio.to_thread(path.unlink, True)  # missing_ok=True

    async def exists(self, key: str) -> bool:
        path = self._resolve(key)
        return await asyncio.to_thread(path.is_file)
