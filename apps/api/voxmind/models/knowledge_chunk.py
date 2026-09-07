"""One chunk of a `KnowledgeDocument`: the unit retrieval operates over.
`embedding` is a real, genuinely-computed sentence embedding (see
services/knowledge/embedding_provider.py) - `pgvector`'s `Vector` type maps
straight to a Postgres `vector(EMBEDDING_DIM)` column, indexed with an
ivfflat approximate-nearest-neighbor index (see the Phase 4 migration).
Lexical search does not use a stored column at all: it matches against
`to_tsvector('english', content)` computed at query time, backed by a
functional GIN index - see repositories/knowledge_chunk_repository.py.
`document_title`/`chunk_index` are denormalized onto every chunk so a
`RetrievedChunk` can carry full citation metadata without an extra join at
read time.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.core.config import get_settings
from voxmind.db.base import Base

_EMBEDDING_DIM = get_settings().EMBEDDING_DIM


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Denormalized from the parent document so retrieval can scope a "
        "single-table search to one conversation without a join.",
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    document_title: Mapped[str] = mapped_column(String(300), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(_EMBEDDING_DIM), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
