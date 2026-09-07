from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MessageCreate(BaseModel):
    """Plain user-authored text only. Assistant messages are never created
    directly through this endpoint - they can only be produced by the real
    RAG pipeline (`POST /conversations/{id}/ask`, Phase 4), never fabricated
    by a client-supplied role."""

    role: Literal["user"]
    content: str = Field(min_length=1, max_length=8000)


class MessageOut(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    audio_asset_id: uuid.UUID | None
    created_at: datetime

    model_config = {"from_attributes": True}
