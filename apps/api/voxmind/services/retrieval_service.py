"""RAG retrieval application service: real embedding, real pgvector cosine
search, real PostgreSQL full-text search, deterministic RRF fusion, and
optional real cross-encoder reranking - composed here because fusion needs
direct database access (KnowledgeChunkRepository), which `PipelineStage`s
never touch (see services/retrieval/stages.py). Every retrieval run is
persisted in full via `RetrievalResultRepository` so it stays inspectable
and reproducible, per the Phase 4 requirement.
"""
from __future__ import annotations

import math
import time
import uuid
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings
from voxmind.repositories.knowledge_chunk_repository import KnowledgeChunkRepository
from voxmind.repositories.retrieval_result_repository import RetrievalResultRepository
from voxmind.services.knowledge.embedding_provider import HuggingFaceEmbeddingProvider
from voxmind.services.knowledge.stages import ChunkEmbeddingInput, ChunkEmbeddingOutput, ChunkEmbeddingStage
from voxmind.services.retrieval.hybrid import FusedCandidate, reciprocal_rank_fusion
from voxmind.services.retrieval.interfaces import RetrievedChunk
from voxmind.services.retrieval.reranker import CrossEncoderReranker
from voxmind.services.retrieval.stages import RerankInput, RerankOutput, RerankStage
from voxmind.workers.task_runner import JobStatus, PipelineStage, TaskRunner, wait_for_completion


class RetrievalService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        settings: Settings,
        task_runner: TaskRunner,
        embedding_provider: HuggingFaceEmbeddingProvider,
        reranker: CrossEncoderReranker,
    ) -> None:
        self._session = session
        self._settings = settings
        self._task_runner = task_runner
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._chunks = KnowledgeChunkRepository(session)
        self._results = RetrievalResultRepository(session)

    async def retrieve(
        self, *, conversation_id: uuid.UUID, query_message_id: uuid.UUID, query: str
    ) -> tuple[list[RetrievedChunk], uuid.UUID]:
        settings = self._settings
        started = time.monotonic()

        total_chunks = await self._chunks.count_for_session(conversation_id)
        if total_chunks == 0:
            result_row = await self._results.create(
                session_id=conversation_id,
                query_message_id=query_message_id,
                query_text=query,
                embedding_model=self._embedding_provider.model_id,
                top_k=settings.RETRIEVAL_TOP_K,
                reranked=False,
                items=[],
                latency_ms=round((time.monotonic() - started) * 1000),
            )
            await self._session.commit()
            return [], result_row.id

        embed_handle = await self._task_runner.dispatch(
            cast(PipelineStage, ChunkEmbeddingStage(self._embedding_provider)),
            ChunkEmbeddingInput(text=query),
        )
        embed_status = await wait_for_completion(self._task_runner, embed_handle.job_id)
        if embed_status.status == JobStatus.FAILED:
            result_row = await self._results.create(
                session_id=conversation_id,
                query_message_id=query_message_id,
                query_text=query,
                embedding_model=self._embedding_provider.model_id,
                top_k=settings.RETRIEVAL_TOP_K,
                reranked=False,
                items=[],
                latency_ms=round((time.monotonic() - started) * 1000),
            )
            await self._session.commit()
            return [], result_row.id
        embed_result = cast(ChunkEmbeddingOutput, await self._task_runner.get_result(embed_handle.job_id))

        vector_results = await self._chunks.search_vector(
            session_id=conversation_id,
            embedding=embed_result.embedding,
            top_k=settings.RETRIEVAL_CANDIDATE_K,
        )
        lexical_results = await self._chunks.search_lexical(
            session_id=conversation_id, query=query, top_k=settings.RETRIEVAL_CANDIDATE_K
        )
        fused = reciprocal_rank_fusion(
            vector_results=vector_results, lexical_results=lexical_results, rrf_k=settings.HYBRID_RRF_K
        )

        reranked = False
        rerank_scores: dict[uuid.UUID, float] = {}
        candidates_for_rerank = fused[: settings.RETRIEVAL_CANDIDATE_K]
        if settings.RERANK_ENABLED and candidates_for_rerank:
            try:
                rerank_handle = await self._task_runner.dispatch(
                    cast(PipelineStage, RerankStage(self._reranker)),
                    RerankInput(
                        query=query, passages=[c.chunk.content for c in candidates_for_rerank]
                    ),
                )
                rerank_status = await wait_for_completion(self._task_runner, rerank_handle.job_id)
                if rerank_status.status != JobStatus.FAILED:
                    rerank_output = cast(
                        RerankOutput, await self._task_runner.get_result(rerank_handle.job_id)
                    )
                    for candidate, score in zip(candidates_for_rerank, rerank_output.scores, strict=True):
                        # NaN/Infinity are valid Python floats but invalid
                        # JSON - would otherwise fail to persist into JSONB.
                        if math.isfinite(score):
                            rerank_scores[candidate.chunk.id] = score
                    reranked = True
            except Exception:  # noqa: BLE001 - reranking is an optimization, never fatal to retrieval
                reranked = False

        if reranked:
            ordered = sorted(
                candidates_for_rerank, key=lambda c: rerank_scores.get(c.chunk.id, float("-inf")), reverse=True
            )
        else:
            ordered = fused

        top_k = ordered[: settings.RETRIEVAL_TOP_K]
        selected_ids = {c.chunk.id for c in top_k}

        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=c.chunk.id,
                document_id=c.chunk.document_id,
                text=c.chunk.content,
                citation=f"{c.chunk.document_title} (chunk {c.chunk.chunk_index + 1})",
                vector_score=c.vector_score,
                lexical_score=c.lexical_score,
                rerank_score=rerank_scores.get(c.chunk.id),
            )
            for c in top_k
        ]

        items = [_trace_item(c, rerank_scores, rank, selected_ids) for rank, c in enumerate(fused, start=1)]
        result_row = await self._results.create(
            session_id=conversation_id,
            query_message_id=query_message_id,
            query_text=query,
            embedding_model=self._embedding_provider.model_id,
            top_k=settings.RETRIEVAL_TOP_K,
            reranked=reranked,
            items=items,
            latency_ms=round((time.monotonic() - started) * 1000),
        )
        await self._session.commit()
        return retrieved_chunks, result_row.id


def _trace_item(
    candidate: FusedCandidate,
    rerank_scores: dict[uuid.UUID, float],
    hybrid_rank: int,
    selected_ids: set[uuid.UUID],
) -> dict:
    return {
        "chunk_id": str(candidate.chunk.id),
        "document_id": str(candidate.chunk.document_id),
        "document_title": candidate.chunk.document_title,
        "chunk_index": candidate.chunk.chunk_index,
        "excerpt": candidate.chunk.content[:300],
        "vector_score": candidate.vector_score,
        "vector_rank": candidate.vector_rank,
        "lexical_score": candidate.lexical_score,
        "lexical_rank": candidate.lexical_rank,
        "hybrid_score": candidate.hybrid_score,
        "hybrid_rank": hybrid_rank,
        "rerank_score": rerank_scores.get(candidate.chunk.id),
        "selected": candidate.chunk.id in selected_ids,
    }
