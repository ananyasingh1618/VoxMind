from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class VoiceTurnOut(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    user_message_id: uuid.UUID | None
    assistant_message_id: uuid.UUID | None
    llm_generation_id: uuid.UUID | None
    status: str
    error_code: str | None
    error_message: str | None
    audio_content_type: str | None
    audio_provider: str | None
    audio_duration_ms: int | None
    stage_latencies_ms: dict
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}
