from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import (
    get_current_user,
    get_embedding_provider,
    get_llm_provider,
    get_moderation_provider,
    get_nlp_analyzer,
    get_reranker,
    get_settings,
    get_task_runner,
)
from voxmind.core.config import Settings
from voxmind.core.rate_limit import rate_limit_by_user
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.message import MessageOut
from voxmind.schemas.rag import AskRequest, AskResponse, GenerationOut, GuardrailInfoOut, RetrievedChunkOut
from voxmind.services.guardrail_service import GuardrailService
from voxmind.services.guardrails.interfaces import ModerationProvider
from voxmind.services.knowledge.embedding_provider import HuggingFaceEmbeddingProvider
from voxmind.services.llm.interfaces import LlmProvider
from voxmind.services.memory_service import MemoryService
from voxmind.services.nlp.analyzer import RealNlpAnalyzer
from voxmind.services.nlp_service import NlpService
from voxmind.services.rag_service import RagService
from voxmind.services.retrieval.reranker import CrossEncoderReranker
from voxmind.services.retrieval_service import RetrievalService
from voxmind.workers.task_runner import TaskRunner

router = APIRouter(prefix="/conversations/{conversation_id}", tags=["rag"])


def get_rag_service(
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    task_runner: TaskRunner = Depends(get_task_runner),
    llm_provider: LlmProvider | None = Depends(get_llm_provider),
    nlp_analyzer: RealNlpAnalyzer = Depends(get_nlp_analyzer),
    embedding_provider: HuggingFaceEmbeddingProvider = Depends(get_embedding_provider),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    moderation_provider: ModerationProvider = Depends(get_moderation_provider),
) -> RagService:
    nlp_service = NlpService(session, task_runner=task_runner, analyzer=nlp_analyzer)
    memory_service = MemoryService(session, settings=settings, llm_provider=llm_provider)
    retrieval_service = RetrievalService(
        session,
        settings=settings,
        task_runner=task_runner,
        embedding_provider=embedding_provider,
        reranker=reranker,
    )
    guardrail_service = GuardrailService(session, moderation_provider=moderation_provider)
    return RagService(
        session,
        settings=settings,
        llm_provider=llm_provider,
        nlp_service=nlp_service,
        memory_service=memory_service,
        retrieval_service=retrieval_service,
        guardrail_service=guardrail_service,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=201,
    dependencies=[Depends(rate_limit_by_user("expensive", "RATE_LIMIT_EXPENSIVE_PER_WINDOW"))],
)
async def ask(
    conversation_id: uuid.UUID,
    body: AskRequest,
    current_user: User = Depends(get_current_user),
    service: RagService = Depends(get_rag_service),
) -> AskResponse:
    """The full Phase 4 RAG flow: creates the user message, runs NLP, loads
    memory, retrieves knowledge (vector + lexical + hybrid fusion + optional
    rerank), assembles a token-budgeted context, and calls the configured
    LLM provider. If no provider is configured, `generation.grounding_status`
    is `"unavailable"` and no assistant message is created - never a
    fabricated answer. See docs/rag.md.
    """
    result = await service.ask(
        conversation_id=conversation_id, user_id=current_user.id, question=body.question
    )
    guardrail = result.guardrail_result
    return AskResponse(
        user_message=MessageOut.model_validate(result.user_message),
        assistant_message=(
            MessageOut.model_validate(result.assistant_message) if result.assistant_message else None
        ),
        retrieval_result_id=result.retrieval_result_id,
        retrieved_chunks=[RetrievedChunkOut(**c.model_dump()) for c in result.retrieved_chunks],
        generation=GenerationOut.model_validate(result.generation),
        guardrail=(
            GuardrailInfoOut(
                decision=guardrail.decision.value,
                reasons=guardrail.reasons,
                filtered_chunk_ids=guardrail.filtered_chunk_ids,
                safety_flagged=guardrail.safety.is_unsafe,
                safety_categories=guardrail.safety.categories,
            )
            if guardrail
            else None
        ),
    )
