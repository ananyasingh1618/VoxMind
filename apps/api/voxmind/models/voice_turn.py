"""One full pass through the Phase 5 real-time voice loop (audio capture →
STT → analysis → memory/RAG → LLM → TTS). Tracks the lifecycle of a single
conversational turn end-to-end, and persists genuinely measured per-stage
latencies (`stage_latencies_ms`) - never estimated - so the loop's real
performance characteristics are inspectable after the fact, exactly like
`AudioProcessingJob.stage_durations_ms`/`EmotionProcessingJob.stage_durations_ms`.

`status="interrupted"` is a real, distinct outcome (the client disconnected
mid-pipeline, detected via `Request.is_disconnected()` between stages -
see services/voice_turn_service.py) - not merged into "failed", since an
interruption is expected user behavior, not an error.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base

VALID_STATUSES = ("pending", "completed", "partial", "failed", "interrupted")


class VoiceTurn(Base):
    __tablename__ = "voice_turns"
    __table_args__ = (CheckConstraint(f"status IN {VALID_STATUSES}", name="valid_status"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("turns.id", ondelete="SET NULL"), nullable=True
    )
    assistant_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("turns.id", ondelete="SET NULL"), nullable=True
    )
    llm_generation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("llm_generations.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_storage_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    audio_content_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    audio_provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    audio_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stage_latencies_ms: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Real, measured durations: stt_ms, analysis_ms, retrieval_ms, llm_ms, tts_ms, total_ms.",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
