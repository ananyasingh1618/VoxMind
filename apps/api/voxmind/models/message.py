"""Message model.

Table name is `turns` (matching the approved Phase 0 schema's entity #3 —
one user or assistant utterance within a session), exposed to the app as
`Message`. `content` is the equivalent of the blueprint's `raw_text` column,
renamed for clarity since Phase 1 has no STT yet and messages are plain
persisted text; this is a naming refinement only, not a schema redesign —
noted explicitly in the Phase 1 report as an implementation decision.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from voxmind.db.base import Base

if TYPE_CHECKING:
    from voxmind.models.conversation import Conversation

VALID_ROLES = ("user", "assistant")


class Message(Base):
    __tablename__ = "turns"
    __table_args__ = (CheckConstraint(f"role IN {VALID_ROLES}", name="valid_role"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    audio_asset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("audio_assets.id", ondelete="SET NULL"), nullable=True
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
