from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.retrieval_result import RetrievalResult


class RetrievalResultRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        session_id: uuid.UUID,
        query_message_id: uuid.UUID,
        query_text: str,
        embedding_model: str,
        top_k: int,
        reranked: bool,
        items: list[dict],
        latency_ms: int = 0,
    ) -> RetrievalResult:
        row = RetrievalResult(
            session_id=session_id,
            query_message_id=query_message_id,
            query_text=query_text,
            embedding_model=embedding_model,
            top_k=top_k,
            reranked=reranked,
            items=items,
            latency_ms=latency_ms,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_by_id(self, retrieval_result_id: uuid.UUID) -> RetrievalResult | None:
        return await self._session.get(RetrievalResult, retrieval_result_id)
