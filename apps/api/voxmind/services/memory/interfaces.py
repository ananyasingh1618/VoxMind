"""Conversation memory service boundary. Implemented in Phase 4."""
from __future__ import annotations

import uuid
from typing import Protocol

from pydantic import BaseModel


class ConversationSummary(BaseModel):
    id: uuid.UUID
    summary_text: str
    covers_message_ids: list[uuid.UUID]


class MemoryContext(BaseModel):
    recent_message_ids: list[uuid.UUID]
    relevant_summary: ConversationSummary | None


class MemoryProvider(Protocol):
    async def get_context(self, conversation_id: uuid.UUID) -> MemoryContext: ...

    async def summarize_if_needed(self, conversation_id: uuid.UUID) -> ConversationSummary | None: ...
