"""Response shape for `POST /conversations/{id}/ask`, structured so the UI
can render the full pipeline trace requested by Phase 4: question -> ranked
retrieved evidence -> generated answer -> citations -> grounding status ->
(Phase 6) the guardrail decision that produced the final answer.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from voxmind.schemas.message import MessageOut


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class CitationOut(BaseModel):
    chunk_id: str
    document_title: str
    excerpt: str


class RetrievedChunkOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    citation: str
    vector_score: float | None
    lexical_score: float | None
    rerank_score: float | None


class GenerationOut(BaseModel):
    id: uuid.UUID
    provider: str
    model_name: str | None
    context_token_estimate: int
    answer: str
    raw_model_answer: str
    citations: list[CitationOut]
    confidence: float
    evidence_summary: str
    grounding_status: str
    grounding_details: dict
    error_message: str | None
    latency_ms: int
    created_at: datetime

    model_config = {"from_attributes": True}


class GuardrailInfoOut(BaseModel):
    decision: str
    reasons: list[str]
    filtered_chunk_ids: list[str]
    safety_flagged: bool
    safety_categories: list[str]


class AskResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut | None
    retrieval_result_id: uuid.UUID
    retrieved_chunks: list[RetrievedChunkOut]
    generation: GenerationOut
    guardrail: GuardrailInfoOut | None = None
