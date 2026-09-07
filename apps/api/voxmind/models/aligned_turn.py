"""Derived, speaker-attributed turns produced by the deterministic alignment
algorithm (voxmind/services/speech/alignment.py) - explicitly distinct from
the raw `transcript_segments`/`speaker_segments` model output it's derived
from. `speaker_label` is nullable: a turn with no overlapping diarization
segment is left unattributed rather than guessed.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base


class AlignedTurn(Base):
    __tablename__ = "aligned_turns"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("turns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    speaker_label: Mapped[str | None] = mapped_column(String(50), nullable=True)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
