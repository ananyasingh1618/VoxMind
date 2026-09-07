from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class RecentTurnOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationSummaryOut(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    summary_text: str
    covers_message_ids: list[str]
    method: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationMemoryOut(BaseModel):
    recent_turns: list[RecentTurnOut]
    latest_summary: ConversationSummaryOut | None
