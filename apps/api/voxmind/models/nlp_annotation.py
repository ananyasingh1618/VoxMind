"""Real NLP annotation of a message's text: sentiment, intent, topics, and
named entities - one row per message, produced by `NlpService` (real HF
sentiment/NER models plus deterministic topic/intent extraction; see
docs/nlp.md). This is the "semantic signal" half of Phase 4's incongruence
analysis - `IncongruenceSignal.semantic_signal` is derived from this table's
`sentiment_label`/`sentiment_score`, never re-computed ad hoc.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from voxmind.db.base import Base


class NlpAnnotation(Base):
    __tablename__ = "nlp_annotations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("turns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sentiment_label: Mapped[str] = mapped_column(String(20), nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)
    intent_label: Mapped[str] = mapped_column(String(50), nullable=False)
    intent_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    topics: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    entities: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    model_versions: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, doc="e.g. {'sentiment': '<model id>', 'ner': '<model id>'}"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
