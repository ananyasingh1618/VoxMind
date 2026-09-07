"""RAG retrieval service boundary. Implemented in Phase 4.

`VectorStore` is intentionally separate from `RetrievalProvider`: the vector
store is a swappable low-level index (pgvector today, a dedicated vector DB
later if evaluation shows it's warranted); the retrieval provider composes
vector + lexical search + optional reranking on top of it.
"""
from __future__ import annotations

import uuid
from typing import Protocol

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    citation: str
    vector_score: float | None
    lexical_score: float | None
    rerank_score: float | None


class VectorStore(Protocol):
    async def add(self, chunk_id: uuid.UUID, embedding: list[float], metadata: dict) -> None: ...

    async def search(self, embedding: list[float], top_k: int) -> list[tuple[uuid.UUID, float]]: ...

    async def delete(self, chunk_id: uuid.UUID) -> None: ...


class RetrievalProvider(Protocol):
    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...
