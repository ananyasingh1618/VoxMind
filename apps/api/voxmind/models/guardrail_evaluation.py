"""One guardrail decision per LLM generation - the persisted record behind
"guardrail decision logging". Every real `LlmResponse` that comes out of
`services/rag_service.py::generate_for_message()` passes through exactly
one of these before it can become a persisted assistant `Message` or reach
TTS; `decision` is the only outcome that ever leaves the guardrail layer.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base

VALID_DECISIONS = ("approved", "modified", "blocked")


class GuardrailEvaluation(Base):
    __tablename__ = "guardrail_evaluations"
    __table_args__ = (CheckConstraint(f"decision IN {VALID_DECISIONS}", name="valid_decision"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    llm_generation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("llm_generations.id", ondelete="SET NULL"), nullable=True
    )
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    reasons: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    input_injection_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    input_injection_patterns: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    filtered_chunk_ids: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, doc="Retrieved chunk ids excluded from context by injection filtering."
    )
    output_validation_issues: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    safety_flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    safety_categories: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    safety_method: Mapped[str] = mapped_column(String(30), nullable=False)
    original_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    final_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
