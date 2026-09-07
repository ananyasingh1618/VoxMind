from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str | None
    status: str
    created_at: datetime
    ended_at: datetime | None

    model_config = {"from_attributes": True}
