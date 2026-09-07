from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class EntityOut(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int


class NlpAnnotationOut(BaseModel):
    id: uuid.UUID
    message_id: uuid.UUID
    sentiment_label: str
    sentiment_score: float
    intent_label: str
    intent_confidence: float
    topics: list[str]
    entities: list[dict]
    model_versions: dict[str, str]
    created_at: datetime

    model_config = {"from_attributes": True}
