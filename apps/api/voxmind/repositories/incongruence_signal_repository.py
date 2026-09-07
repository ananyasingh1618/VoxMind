from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.aligned_turn import AlignedTurn
from voxmind.models.incongruence_signal import IncongruenceSignal
from voxmind.services.incongruence.interfaces import IncongruenceSignal as IncongruenceResult


class IncongruenceSignalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, aligned_turn_id: uuid.UUID, result: IncongruenceResult
    ) -> IncongruenceSignal:
        row = IncongruenceSignal(
            aligned_turn_id=aligned_turn_id,
            incongruence_score=result.incongruence_score,
            semantic_signal=result.semantic_signal,
            vocal_signal=result.vocal_signal,
            confidence=result.confidence,
            explanation=result.explanation,
            signal_category=result.signal_category,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_for_message(self, message_id: uuid.UUID) -> list[IncongruenceSignal]:
        stmt = (
            select(IncongruenceSignal)
            .join(AlignedTurn, IncongruenceSignal.aligned_turn_id == AlignedTurn.id)
            .where(AlignedTurn.message_id == message_id)
            .order_by(AlignedTurn.start_ms.asc())
        )
        return list((await self._session.execute(stmt)).scalars().all())
