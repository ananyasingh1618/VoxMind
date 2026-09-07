"""One model prediction for one aligned speaker turn.

Deliberately turn-scoped, not message- or conversation-scoped: emotion is
predicted per `AlignedTurn` (Phase 2's speaker-attributed segment), never
claimed for an entire recording or for a speaker who wasn't isolated by
diarization. `probabilities` is the full distribution the model actually
output - `predicted_label` is redundant with it (the argmax) but kept as an
indexed column for cheap querying.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base


class EmotionPrediction(Base):
    __tablename__ = "emotion_predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    aligned_turn_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("aligned_turns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    predicted_label: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, doc="max(probabilities.values())")
    probabilities: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
