"""Long-term conversation memory: a rolling summary that replaces raw older
messages in the LLM's context once a conversation grows past
`Settings.SUMMARY_TRIGGER_MESSAGE_COUNT` un-summarized messages. Each row is
immutable once created - a new summarization pass inserts a new row covering
the newly-accrued messages (plus, in its input text, the previous summary),
so the full history of what the assistant "knew" at each point stays
auditable rather than being overwritten in place.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    covers_message_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    method: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        doc="'llm_generated' when a configured LLM provider produced it, "
        "'extractive_fallback' when no provider was available.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
