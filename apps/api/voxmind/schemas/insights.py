"""Conversation insights response shapes (Phase 6). `observations` are
factual, model-attributed statements about what was actually detected
(sentiment/emotion/incongruence labels and counts, with the model that
produced them named). `interpretations` are explicitly hedged, non-
diagnostic readings of a real pattern in those observations - every one
carries the same standard caveat and is never generated unless a real
pattern exists in the data. Neither ever uses psychological or medical
diagnostic language (see services/insights_service.py).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel

NON_DIAGNOSTIC_CAVEAT = (
    "This is a descriptive, pattern-based observation - not a psychological or medical assessment."
)


class TimelineEmotionEntry(BaseModel):
    speaker_label: str | None
    predicted_label: str
    confidence: float


class TimelineIncongruenceEntry(BaseModel):
    incongruence_score: float
    confidence: float
    explanation: str


class TimelineEntry(BaseModel):
    message_id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    sentiment_label: str | None
    sentiment_score: float | None
    intent_label: str | None
    topics: list[str]
    emotion: list[TimelineEmotionEntry]
    incongruence: list[TimelineIncongruenceEntry]


class Observation(BaseModel):
    text: str
    basis: str


class Interpretation(BaseModel):
    text: str
    caveat: str = NON_DIAGNOSTIC_CAVEAT


class ConversationInsights(BaseModel):
    conversation_id: uuid.UUID
    message_count: int
    timeline: list[TimelineEntry]
    observations: list[Observation]
    interpretations: list[Interpretation]
