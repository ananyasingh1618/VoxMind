from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class IncongruenceSignalOut(BaseModel):
    id: uuid.UUID
    aligned_turn_id: uuid.UUID
    incongruence_score: float
    semantic_signal: dict
    vocal_signal: dict
    confidence: float
    explanation: str
    signal_category: str
    created_at: datetime

    model_config = {"from_attributes": True}
