from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.nlp_annotation import NlpAnnotation
from voxmind.services.nlp.interfaces import NlpAnnotation as NlpAnnotationResult


class NlpAnnotationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, *, message_id: uuid.UUID, result: NlpAnnotationResult) -> NlpAnnotation:
        row = NlpAnnotation(
            message_id=message_id,
            sentiment_label=result.sentiment_label,
            sentiment_score=result.sentiment_score,
            intent_label=result.intent_label,
            intent_confidence=result.intent_confidence,
            topics=result.topics,
            entities=[entity.model_dump() for entity in result.entities],
            model_versions=result.model_versions,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_for_message(self, message_id: uuid.UUID) -> NlpAnnotation | None:
        stmt = (
            select(NlpAnnotation)
            .where(NlpAnnotation.message_id == message_id)
            .order_by(NlpAnnotation.created_at.desc())
            .limit(1)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()
