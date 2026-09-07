from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.llm_generation import LlmGeneration


class LlmGenerationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, **fields: object) -> LlmGeneration:
        row = LlmGeneration(**fields)
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_by_id(self, generation_id: uuid.UUID) -> LlmGeneration | None:
        return await self._session.get(LlmGeneration, generation_id)

    async def get_for_answer_message(self, answer_message_id: uuid.UUID) -> LlmGeneration | None:
        stmt = select(LlmGeneration).where(LlmGeneration.answer_message_id == answer_message_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()
