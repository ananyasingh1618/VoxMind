from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.conversation_summary import ConversationSummary


class ConversationSummaryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        session_id: uuid.UUID,
        summary_text: str,
        covers_message_ids: list[uuid.UUID],
        method: str,
    ) -> ConversationSummary:
        row = ConversationSummary(
            session_id=session_id,
            summary_text=summary_text,
            covers_message_ids=[str(mid) for mid in covers_message_ids],
            method=method,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_latest(self, session_id: uuid.UUID) -> ConversationSummary | None:
        stmt = (
            select(ConversationSummary)
            .where(ConversationSummary.session_id == session_id)
            .order_by(ConversationSummary.created_at.desc())
            .limit(1)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list_for_session(self, session_id: uuid.UUID) -> list[ConversationSummary]:
        stmt = (
            select(ConversationSummary)
            .where(ConversationSummary.session_id == session_id)
            .order_by(ConversationSummary.created_at.asc())
        )
        return list((await self._session.execute(stmt)).scalars().all())
