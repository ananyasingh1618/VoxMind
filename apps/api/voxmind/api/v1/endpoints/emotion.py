from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import (
    get_acoustic_feature_extractor,
    get_current_user,
    get_storage_backend,
    get_task_runner,
    get_wav2vec2_provider,
)
from voxmind.core.rate_limit import rate_limit_by_user
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.emotion import EmotionPredictionOut, EmotionProcessingJobOut
from voxmind.services.emotion.acoustic_features import LibrosaAcousticFeatureExtractor
from voxmind.services.emotion.wav2vec_provider import HuggingFaceWav2Vec2Provider
from voxmind.services.emotion_service import EmotionService
from voxmind.services.storage.interfaces import StorageBackend
from voxmind.workers.task_runner import TaskRunner

router = APIRouter(prefix="/conversations/{conversation_id}", tags=["emotion"])


def get_emotion_service(
    session: AsyncSession = Depends(get_db),
    storage: StorageBackend = Depends(get_storage_backend),
    task_runner: TaskRunner = Depends(get_task_runner),
    acoustic_extractor: LibrosaAcousticFeatureExtractor = Depends(get_acoustic_feature_extractor),
    embedding_provider: HuggingFaceWav2Vec2Provider = Depends(get_wav2vec2_provider),
) -> EmotionService:
    return EmotionService(
        session,
        storage=storage,
        task_runner=task_runner,
        acoustic_extractor=acoustic_extractor,
        embedding_provider=embedding_provider,
    )


@router.post(
    "/messages/{message_id}/emotion/process",
    response_model=EmotionProcessingJobOut,
    status_code=201,
    dependencies=[Depends(rate_limit_by_user("expensive", "RATE_LIMIT_EXPENSIVE_PER_WINDOW"))],
)
async def process_emotion(
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: EmotionService = Depends(get_emotion_service),
) -> EmotionProcessingJobOut:
    """Runs emotion inference over every aligned turn in this message.
    Genuinely executes synchronously under `InProcessTaskRunner`, same
    job-API-shaped contract as Phase 2's `/audio/.../process`. If no
    trained, active emotion model is registered, `status` is
    `"unavailable"` (not a fabricated prediction) - see docs/emotion.md.
    """
    job = await service.process_message(
        conversation_id=conversation_id, user_id=current_user.id, message_id=message_id
    )
    return EmotionProcessingJobOut.model_validate(job)


@router.get("/emotion-jobs/{job_id}", response_model=EmotionProcessingJobOut)
async def get_emotion_job(
    conversation_id: uuid.UUID,
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: EmotionService = Depends(get_emotion_service),
) -> EmotionProcessingJobOut:
    job = await service.get_job(conversation_id=conversation_id, user_id=current_user.id, job_id=job_id)
    return EmotionProcessingJobOut.model_validate(job)


@router.get("/messages/{message_id}/emotion", response_model=list[EmotionPredictionOut])
async def list_emotion_predictions(
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: EmotionService = Depends(get_emotion_service),
) -> list[EmotionPredictionOut]:
    predictions = await service.get_predictions_for_message(
        conversation_id=conversation_id, user_id=current_user.id, message_id=message_id
    )
    return [EmotionPredictionOut.model_validate(p) for p in predictions]
