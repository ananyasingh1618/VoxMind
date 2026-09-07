"""Semantic-vocal incongruence signal for one aligned (speaker-attributed)
turn: a deterministic comparison of *what* was said (semantic signal, from
`NlpAnnotation`) against *how* it was said (vocal signal, from
`EmotionPrediction`). `signal_category` is a machine-readable guardrail
against this drifting into deception/lie-detector framing - see
services/incongruence/interfaces.py and docs/DECISIONS/0002 for why.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base

SIGNAL_CATEGORY = "analytical_not_diagnostic"


class IncongruenceSignal(Base):
    __tablename__ = "incongruence_signals"
    __table_args__ = (
        CheckConstraint(f"signal_category = '{SIGNAL_CATEGORY}'", name="valid_signal_category"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    aligned_turn_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("aligned_turns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    incongruence_score: Mapped[float] = mapped_column(Float, nullable=False)
    semantic_signal: Mapped[dict] = mapped_column(JSONB, nullable=False)
    vocal_signal: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    signal_category: Mapped[str] = mapped_column(String(30), nullable=False, default=SIGNAL_CATEGORY)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
