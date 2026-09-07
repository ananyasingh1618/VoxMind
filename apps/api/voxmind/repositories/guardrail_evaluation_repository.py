from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.guardrail_evaluation import GuardrailEvaluation


class GuardrailEvaluationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, **fields: object) -> GuardrailEvaluation:
        row = GuardrailEvaluation(**fields)
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_for_session(self, session_id: uuid.UUID) -> list[GuardrailEvaluation]:
        stmt = (
            select(GuardrailEvaluation)
            .where(GuardrailEvaluation.session_id == session_id)
            .order_by(GuardrailEvaluation.created_at.asc())
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def count_by_decision_for_user(self, user_id: uuid.UUID) -> dict[str, int]:
        from voxmind.models.conversation import Conversation

        stmt = (
            select(GuardrailEvaluation.decision, func.count())
            .join(Conversation, GuardrailEvaluation.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
            .group_by(GuardrailEvaluation.decision)
        )
        rows = (await self._session.execute(stmt)).all()
        return {decision: count for decision, count in rows}
