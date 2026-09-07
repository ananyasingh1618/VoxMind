"""Audio asset model.

Phase 2 note: an upload produces one `kind="original"` row (the untouched
uploaded bytes). Real preprocessing (voxmind/services/speech/preprocessing.py)
never overwrites that row - it produces a *separate* `kind="processed"` row
pointing back at the original via `source_asset_id`, holding the canonical
mono/16kHz/PCM16 WAV actually fed to Whisper/diarization. Both rows' audio
bytes live in the storage abstraction (local filesystem or S3/MinIO) - never
in this table, which only ever holds a storage key.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base

VALID_KINDS = ("original", "processed")


class AudioAsset(Base):
    __tablename__ = "audio_assets"
    __table_args__ = (CheckConstraint(f"kind IN {VALID_KINDS}", name="valid_kind"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default="original")
    source_asset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("audio_assets.id", ondelete="CASCADE"), nullable=True
    )
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sample_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channels: Mapped[int | None] = mapped_column(Integer, nullable=True)
    format: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
