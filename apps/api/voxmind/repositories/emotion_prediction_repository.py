from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.emotion_prediction import EmotionPrediction
from voxmind.services.emotion.interfaces import EmotionPrediction as EmotionPredictionResult


class EmotionPredictionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, aligned_turn_id: uuid.UUID, model_version_id: uuid.UUID, result: EmotionPredictionResult
    ) -> EmotionPrediction:
        row = EmotionPrediction(
            aligned_turn_id=aligned_turn_id,
            model_version_id=model_version_id,
            predicted_label=result.label,
            confidence=max(result.probabilities.values()),
            probabilities=result.probabilities,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_for_message(self, message_id: uuid.UUID) -> list[EmotionPrediction]:
        from voxmind.models.aligned_turn import AlignedTurn

        stmt = (
            select(EmotionPrediction)
            .join(AlignedTurn, EmotionPrediction.aligned_turn_id == AlignedTurn.id)
            .where(AlignedTurn.message_id == message_id)
            .order_by(AlignedTurn.start_ms.asc())
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def list_for_turn(self, aligned_turn_id: uuid.UUID) -> list[EmotionPrediction]:
        stmt = (
            select(EmotionPrediction)
            .where(EmotionPrediction.aligned_turn_id == aligned_turn_id)
            .order_by(EmotionPrediction.created_at.desc())
        )
        return list((await self._session.execute(stmt)).scalars().all())
