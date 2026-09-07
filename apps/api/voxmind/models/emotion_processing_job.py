"""Tracks one attempt at running emotion inference over a message's aligned
turns. Mirrors Phase 2's `AudioProcessingJob` pattern deliberately - same
shape, same reasoning: `status="unavailable"` is a genuine, expected
outcome (no trained+active emotion model is registered yet in this
environment) and is tracked distinctly from `"failed"` (the job ran but hit
a genuine error), exactly as Phase 2 distinguished `diarization_status`
"unavailable" from "failed". See docs/emotion.md and
docs/DECISIONS/0005-emotion-requires-trained-model.md.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base

VALID_STATUSES = ("pending", "running", "completed", "failed", "unavailable")


class EmotionProcessingJob(Base):
    __tablename__ = "emotion_processing_jobs"
    __table_args__ = (CheckConstraint(f"status IN {VALID_STATUSES}", name="valid_status"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("turns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    turns_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stage_durations_ms: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
