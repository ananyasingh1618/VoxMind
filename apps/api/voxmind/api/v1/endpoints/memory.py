from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import get_current_user, get_llm_provider, get_settings
from voxmind.core.config import Settings
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.memory import ConversationMemoryOut, ConversationSummaryOut, RecentTurnOut
from voxmind.services.conversation_service import ConversationService
from voxmind.services.llm.interfaces import LlmProvider
from voxmind.services.memory_service import MemoryService

router = APIRouter(prefix="/conversations/{conversation_id}", tags=["memory"])


def get_memory_service(
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    llm_provider: LlmProvider | None = Depends(get_llm_provider),
) -> MemoryService:
    return MemoryService(session, settings=settings, llm_provider=llm_provider)


@router.get("/memory", response_model=ConversationMemoryOut)
async def get_memory(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
) -> ConversationMemoryOut:
    await ConversationService(session).get_owned(conversation_id=conversation_id, user_id=current_user.id)
    recent_turns = await service.get_recent_turns(conversation_id)
    latest_summary = await service.get_latest_summary(conversation_id)
    return ConversationMemoryOut(
        recent_turns=[RecentTurnOut.model_validate(m) for m in recent_turns],
        latest_summary=ConversationSummaryOut.model_validate(latest_summary) if latest_summary else None,
    )


@router.post("/memory/summarize", response_model=ConversationSummaryOut | None, status_code=201)
async def summarize_now(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
    service: MemoryService = Depends(get_memory_service),
) -> ConversationSummaryOut | None:
    """Manually triggers the same accrual check `RagService.ask()` runs
    automatically after every answer - useful for inspecting memory behavior
    without generating more turns. Still respects
    `Settings.SUMMARY_TRIGGER_MESSAGE_COUNT`: returns null if fewer
    un-summarized messages have accrued than the threshold."""
    await ConversationService(session).get_owned(conversation_id=conversation_id, user_id=current_user.id)
    summary = await service.summarize_if_needed(conversation_id)
    return ConversationSummaryOut.model_validate(summary) if summary else None
