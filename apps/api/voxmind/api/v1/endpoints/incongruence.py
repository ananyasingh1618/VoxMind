from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import get_current_user, get_incongruence_analyzer, get_nlp_analyzer, get_task_runner
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.incongruence import IncongruenceSignalOut
from voxmind.services.incongruence.analyzer import DeterministicIncongruenceAnalyzer
from voxmind.services.incongruence_service import IncongruenceService
from voxmind.services.nlp.analyzer import RealNlpAnalyzer
from voxmind.services.nlp_service import NlpService
from voxmind.workers.task_runner import TaskRunner

router = APIRouter(prefix="/conversations/{conversation_id}", tags=["incongruence"])


def get_incongruence_service(
    session: AsyncSession = Depends(get_db),
    task_runner: TaskRunner = Depends(get_task_runner),
    nlp_analyzer: RealNlpAnalyzer = Depends(get_nlp_analyzer),
    analyzer: DeterministicIncongruenceAnalyzer = Depends(get_incongruence_analyzer),
) -> IncongruenceService:
    nlp_service = NlpService(session, task_runner=task_runner, analyzer=nlp_analyzer)
    return IncongruenceService(
        session, task_runner=task_runner, nlp_service=nlp_service, analyzer=analyzer
    )


@router.post(
    "/messages/{message_id}/incongruence/analyze",
    response_model=list[IncongruenceSignalOut],
    status_code=201,
)
async def analyze_incongruence(
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: IncongruenceService = Depends(get_incongruence_service),
) -> list[IncongruenceSignalOut]:
    """Compares real per-turn sentiment against real per-turn vocal emotion.
    Turns with no emotion prediction yet are skipped, never fabricated - see
    services/incongruence_service.py."""
    signals = await service.analyze_message(
        conversation_id=conversation_id, user_id=current_user.id, message_id=message_id
    )
    return [IncongruenceSignalOut.model_validate(s) for s in signals]


@router.get("/messages/{message_id}/incongruence", response_model=list[IncongruenceSignalOut])
async def list_incongruence(
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: IncongruenceService = Depends(get_incongruence_service),
) -> list[IncongruenceSignalOut]:
    signals = await service.get_for_message(
        conversation_id=conversation_id, user_id=current_user.id, message_id=message_id
    )
    return [IncongruenceSignalOut.model_validate(s) for s in signals]
