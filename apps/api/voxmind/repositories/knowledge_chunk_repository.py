"""Persistence + the two real, distinct search primitives hybrid retrieval
fuses: `search_vector` (pgvector cosine similarity) and `search_lexical`
(PostgreSQL full-text search - `to_tsvector`/`plainto_tsquery`/`ts_rank_cd`,
not a "BM25-like" approximation). Both return `(chunk, score)` tuples
ordered best-first; fusion happens one layer up, in
`services/retrieval/hybrid.py`, so this repository never needs to know
about ranking strategy.
"""
from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.knowledge_chunk import KnowledgeChunk


class KnowledgeChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        document_id: uuid.UUID,
        session_id: uuid.UUID,
        chunk_index: int,
        document_title: str,
        content: str,
        embedding: list[float],
        embedding_model: str,
    ) -> KnowledgeChunk:
        row = KnowledgeChunk(
            document_id=document_id,
            session_id=session_id,
            chunk_index=chunk_index,
            document_title=document_title,
            content=content,
            char_count=len(content),
            embedding=embedding,
            embedding_model=embedding_model,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_by_id(self, chunk_id: uuid.UUID) -> KnowledgeChunk | None:
        return await self._session.get(KnowledgeChunk, chunk_id)

    async def get_many(self, chunk_ids: list[uuid.UUID]) -> list[KnowledgeChunk]:
        if not chunk_ids:
            return []
        stmt = select(KnowledgeChunk).where(KnowledgeChunk.id.in_(chunk_ids))
        return list((await self._session.execute(stmt)).scalars().all())

    async def search_vector(
        self, *, session_id: uuid.UUID, embedding: list[float], top_k: int
    ) -> list[tuple[KnowledgeChunk, float]]:
        """Cosine-distance nearest-neighbor search via pgvector's `<=>`
        operator (exposed by `pgvector.sqlalchemy`'s `cosine_distance`).
        Returns similarity (1 - distance) so higher is always better,
        matching the convention `search_lexical` uses for ts_rank_cd."""
        distance = KnowledgeChunk.embedding.cosine_distance(embedding)
        stmt = (
            select(KnowledgeChunk, distance.label("distance"))
            .where(KnowledgeChunk.session_id == session_id)
            .order_by(distance.asc())
            .limit(top_k)
        )
        rows = (await self._session.execute(stmt)).all()
        return [(chunk, 1.0 - float(dist)) for chunk, dist in rows]

    async def search_lexical(
        self, *, session_id: uuid.UUID, query: str, top_k: int
    ) -> list[tuple[KnowledgeChunk, float]]:
        """PostgreSQL native full-text search: `to_tsvector('english', content)`
        matched against `plainto_tsquery('english', query)`, ranked with
        `ts_rank_cd` (cover-density ranking - rewards matches where query
        terms appear close together, not just present)."""
        tsvector = func.to_tsvector("english", KnowledgeChunk.content)
        tsquery = func.plainto_tsquery("english", query)
        rank = func.ts_rank_cd(tsvector, tsquery)
        stmt = (
            select(KnowledgeChunk, rank.label("rank"))
            .where(KnowledgeChunk.session_id == session_id, tsvector.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(top_k)
        )
        rows = (await self._session.execute(stmt)).all()
        return [(chunk, float(rank_value)) for chunk, rank_value in rows]

    async def count_for_session(self, session_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(KnowledgeChunk).where(
            KnowledgeChunk.session_id == session_id
        )
        return (await self._session.execute(stmt)).scalar_one()
