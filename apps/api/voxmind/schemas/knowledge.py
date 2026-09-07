from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class KnowledgeDocumentOut(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    title: str
    original_filename: str
    content_type: str
    size_bytes: int
    status: str
    error_message: str | None
    chunk_count: int
    created_at: datetime
    processed_at: datetime | None

    model_config = {"from_attributes": True}
