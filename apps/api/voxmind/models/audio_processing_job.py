"""Tracks one attempt at running the speech pipeline (preprocess → STT →
diarization → alignment) against an uploaded `AudioAsset`.

`status` reflects the *core* pipeline (preprocessing + transcription): if
either fails, `status="failed"` and no transcript exists at all. Diarization
is tracked separately via `diarization_status`, because a missing/invalid
Hugging Face token (an expected, common environment condition - see
docs/audio.md) should not destroy an otherwise-successful transcript. This
job never fabricates a result for a stage that didn't genuinely run.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base

VALID_STATUSES = ("pending", "running", "completed", "failed")
VALID_DIARIZATION_STATUSES = ("completed", "unavailable", "failed")


class AudioProcessingJob(Base):
    __tablename__ = "audio_processing_jobs"
    __table_args__ = (
        CheckConstraint(f"status IN {VALID_STATUSES}", name="valid_status"),
        CheckConstraint(
            f"diarization_status IS NULL OR diarization_status IN {VALID_DIARIZATION_STATUSES}",
            name="valid_diarization_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    audio_asset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("audio_assets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("turns.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    diarization_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    diarization_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_versions: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    stage_durations_ms: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
