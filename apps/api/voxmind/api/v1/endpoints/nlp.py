from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import get_current_user, get_nlp_analyzer, get_task_runner
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.nlp import NlpAnnotationOut
from voxmind.services.nlp.analyzer import RealNlpAnalyzer
from voxmind.services.nlp_service import NlpService
from voxmind.workers.task_runner import TaskRunner

router = APIRouter(prefix="/conversations/{conversation_id}", tags=["nlp"])


def get_nlp_service(
    session: AsyncSession = Depends(get_db),
    task_runner: TaskRunner = Depends(get_task_runner),
    analyzer: RealNlpAnalyzer = Depends(get_nlp_analyzer),
) -> NlpService:
    return NlpService(session, task_runner=task_runner, analyzer=analyzer)


@router.post("/messages/{message_id}/nlp/process", response_model=NlpAnnotationOut, status_code=201)
async def process_nlp(
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: NlpService = Depends(get_nlp_service),
) -> NlpAnnotationOut:
    """Runs real sentiment/NER/topic/intent analysis on a message's text."""
    annotation = await service.process_message(
        conversation_id=conversation_id, user_id=current_user.id, message_id=message_id
    )
    return NlpAnnotationOut.model_validate(annotation)


@router.get("/messages/{message_id}/nlp", response_model=NlpAnnotationOut | None)
async def get_nlp(
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: NlpService = Depends(get_nlp_service),
) -> NlpAnnotationOut | None:
    annotation = await service.get_for_message(
        conversation_id=conversation_id, user_id=current_user.id, message_id=message_id
    )
    return NlpAnnotationOut.model_validate(annotation) if annotation else None
