from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import get_current_user
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.insights import ConversationInsights
from voxmind.services.insights_service import InsightsService

router = APIRouter(prefix="/conversations/{conversation_id}", tags=["insights"])


def get_insights_service(session: AsyncSession = Depends(get_db)) -> InsightsService:
    return InsightsService(session)


@router.get("/insights", response_model=ConversationInsights)
async def get_conversation_insights(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: InsightsService = Depends(get_insights_service),
) -> ConversationInsights:
    """A real, per-message intelligence timeline (sentiment/intent/emotion/
    incongruence, wherever each was actually computed) plus a small set of
    explainable summary insights. `observations` are factual and model-
    attributed; `interpretations` are explicitly hedged and non-diagnostic,
    and only generated when a real pattern supports one - see docs/
    insights.md.
    """
    return await service.get_insights(conversation_id=conversation_id, user_id=current_user.id)
