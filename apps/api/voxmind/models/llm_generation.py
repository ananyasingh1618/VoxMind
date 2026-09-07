"""One structured LLM generation attempt. Only safe, structured fields are
ever persisted here - `answer`, `citations`, `confidence`, `evidence_summary`
- never a model's private chain-of-thought (see
services/llm/interfaces.py and the Phase 0 correction that removed
chain-of-thought fields from the schema entirely).

`grounding_status` is computed by the application after generation, by
checking every citation the model returned against the chunk ids that were
actually retrieved (`retrieval_result_id`) - the model is never trusted to
self-report groundedness. `error_message` is populated (with `answer` left
empty and `grounding_status="unavailable"`) when no LLM provider is
configured or the provider call fails - the same "unavailable/failed, never
fabricated" pattern established in Phase 2/3.

`answer` (Phase 6): after the guardrail layer runs (`services/guardrail_
service.py`), this column is overwritten with the guardrail-approved final
text - the same text the persisted assistant `Message` and TTS receive.
`raw_model_answer` preserves exactly what the LLM originally produced, for
audit purposes only; it is never exposed as "the" answer anywhere in the
API - a client that only ever reads `answer` never sees an unvalidated
response, which is the whole point of the guardrail layer existing.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base

VALID_GROUNDING_STATUSES = ("grounded", "partially_grounded", "ungrounded", "unavailable")


class LlmGeneration(Base):
    __tablename__ = "llm_generations"
    __table_args__ = (
        CheckConstraint(f"grounding_status IN {VALID_GROUNDING_STATUSES}", name="valid_grounding_status"),
    )

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
    answer_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("turns.id", ondelete="SET NULL"),
        nullable=True,
        doc="The assistant Message this generation produced - null when generation failed/was unavailable.",
    )
    retrieval_result_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("retrieval_results.id", ondelete="SET NULL"), nullable=True
    )
    provider: Mapped[str] = mapped_column(String(30), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    context_token_estimate: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    raw_model_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    citations: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    evidence_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    grounding_status: Mapped[str] = mapped_column(String(20), nullable=False)
    grounding_details: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
