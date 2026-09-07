from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.voice_turn import VoiceTurn


class VoiceTurnRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, *, session_id: uuid.UUID) -> VoiceTurn:
        turn = VoiceTurn(session_id=session_id, status="pending")
        self._session.add(turn)
        await self._session.flush()
        return turn

    async def get_by_id(self, turn_id: uuid.UUID) -> VoiceTurn | None:
        return await self._session.get(VoiceTurn, turn_id)

    async def mark_failed(self, turn: VoiceTurn, *, error_code: str, error_message: str) -> None:
        turn.status = "failed"
        turn.error_code = error_code
        turn.error_message = error_message
        turn.completed_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def mark_interrupted(self, turn: VoiceTurn, *, error_message: str) -> None:
        turn.status = "interrupted"
        turn.error_message = error_message
        turn.completed_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def mark_finished(
        self,
        turn: VoiceTurn,
        *,
        status: str,
        user_message_id: uuid.UUID | None,
        assistant_message_id: uuid.UUID | None,
        llm_generation_id: uuid.UUID | None,
        audio_storage_key: str | None,
        audio_content_type: str | None,
        audio_provider: str | None,
        audio_duration_ms: int | None,
        stage_latencies_ms: dict,
        error_message: str | None = None,
    ) -> None:
        turn.status = status
        turn.user_message_id = user_message_id
        turn.assistant_message_id = assistant_message_id
        turn.llm_generation_id = llm_generation_id
        turn.audio_storage_key = audio_storage_key
        turn.audio_content_type = audio_content_type
        turn.audio_provider = audio_provider
        turn.audio_duration_ms = audio_duration_ms
        turn.stage_latencies_ms = stage_latencies_ms
        turn.error_message = error_message
        turn.completed_at = datetime.now(timezone.utc)
        await self._session.flush()
