"""Persisted record of one retrieval run, so RAG answers stay inspectable and
reproducible after the fact rather than only visible in the moment: what was
searched, what came back from vector search, what came back from lexical
search, how they were fused, and (if reranking ran) the final reordering.
`items` preserves source/chunk identity end-to-end - every entry carries its
`chunk_id` and `document_id` even for chunks that were fused but not ranked
into the final top-k, so a retrieval run can be audited in full.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base


class RetrievalResult(Base):
    __tablename__ = "retrieval_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query_message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("turns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    reranked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    items: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="Full fusion trace: [{chunk_id, document_id, document_title, chunk_index, "
        "excerpt, vector_score, vector_rank, lexical_score, lexical_rank, "
        "hybrid_score, hybrid_rank, rerank_score, final_rank, selected}, ...]",
    )
    latency_ms: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Genuinely measured (time.monotonic()) retrieval duration - added in Phase 6 for "
        "analytics/docs/analytics.md; was previously only held transiently on RagAnswer.",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
