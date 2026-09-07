from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import (
    get_current_user,
    get_embedding_provider,
    get_settings,
    get_storage_backend,
    get_task_runner,
)
from voxmind.core.config import Settings
from voxmind.core.rate_limit import rate_limit_by_user
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.knowledge import KnowledgeDocumentOut
from voxmind.services.knowledge.embedding_provider import HuggingFaceEmbeddingProvider
from voxmind.services.knowledge_service import KnowledgeService
from voxmind.services.storage.interfaces import StorageBackend
from voxmind.workers.task_runner import TaskRunner

router = APIRouter(prefix="/conversations/{conversation_id}", tags=["knowledge"])

_knowledge_rate_limit = Depends(rate_limit_by_user("knowledge", "RATE_LIMIT_KNOWLEDGE_PER_WINDOW"))


def get_knowledge_service(
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(get_storage_backend),
    task_runner: TaskRunner = Depends(get_task_runner),
    embedding_provider: HuggingFaceEmbeddingProvider = Depends(get_embedding_provider),
) -> KnowledgeService:
    return KnowledgeService(
        session,
        settings=settings,
        storage=storage,
        task_runner=task_runner,
        embedding_provider=embedding_provider,
    )


@router.post(
    "/documents", response_model=KnowledgeDocumentOut, status_code=201, dependencies=[_knowledge_rate_limit]
)
async def upload_document(
    conversation_id: uuid.UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> KnowledgeDocumentOut:
    raw_bytes = await file.read()
    document = await service.upload(
        conversation_id=conversation_id,
        user_id=current_user.id,
        filename=file.filename or "untitled",
        content_type=file.content_type or "application/octet-stream",
        raw_bytes=raw_bytes,
    )
    return KnowledgeDocumentOut.model_validate(document)


@router.post(
    "/documents/{document_id}/process",
    response_model=KnowledgeDocumentOut,
    status_code=201,
    dependencies=[_knowledge_rate_limit],
)
async def process_document(
    conversation_id: uuid.UUID,
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> KnowledgeDocumentOut:
    """Real parse -> chunk -> embed pipeline, dispatched through TaskRunner.
    Genuinely blocks under `InProcessTaskRunner` for the duration of real
    embedding inference before responding - same job-API-shaped contract as
    Phase 2/3's `.../process` endpoints."""
    document = await service.process(
        conversation_id=conversation_id, user_id=current_user.id, document_id=document_id
    )
    return KnowledgeDocumentOut.model_validate(document)


@router.get("/documents", response_model=list[KnowledgeDocumentOut])
async def list_documents(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> list[KnowledgeDocumentOut]:
    documents = await service.list_for_conversation(conversation_id=conversation_id, user_id=current_user.id)
    return [KnowledgeDocumentOut.model_validate(d) for d in documents]


@router.get("/documents/{document_id}", response_model=KnowledgeDocumentOut)
async def get_document(
    conversation_id: uuid.UUID,
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> KnowledgeDocumentOut:
    document = await service.get(
        conversation_id=conversation_id, user_id=current_user.id, document_id=document_id
    )
    return KnowledgeDocumentOut.model_validate(document)
