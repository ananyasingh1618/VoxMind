from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.audio_asset import AudioAsset


class AudioAssetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        session_id: uuid.UUID,
        storage_key: str,
        kind: str = "original",
        source_asset_id: uuid.UUID | None = None,
        original_filename: str | None = None,
        content_type: str | None = None,
        size_bytes: int | None = None,
        duration_ms: int | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        format: str | None = None,
    ) -> AudioAsset:
        asset = AudioAsset(
            session_id=session_id,
            storage_key=storage_key,
            kind=kind,
            source_asset_id=source_asset_id,
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            channels=channels,
            format=format,
        )
        self._session.add(asset)
        await self._session.flush()
        return asset

    async def get_by_id(self, audio_asset_id: uuid.UUID) -> AudioAsset | None:
        return await self._session.get(AudioAsset, audio_asset_id)

    async def list_for_conversation(self, session_id: uuid.UUID) -> list[AudioAsset]:
        stmt = (
            select(AudioAsset)
            .where(AudioAsset.session_id == session_id, AudioAsset.kind == "original")
            .order_by(AudioAsset.created_at.asc())
        )
        return list((await self._session.execute(stmt)).scalars().all())
