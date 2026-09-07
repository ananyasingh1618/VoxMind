"""LLM generation service boundary. Implemented in Phase 4.

`ConversationContext` is the only thing ever handed to a provider — the LLM
consumes structured signal produced upstream, it never re-derives sentiment/
emotion/retrieval itself. `evidence_summary` replaces any notion of exposing
model chain-of-thought: it is a factual summary of which retrieved facts/
signals the answer relied on, never the model's private reasoning trace.
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class Citation(BaseModel):
    chunk_id: str
    document_title: str
    excerpt: str


class ConversationContext(BaseModel):
    transcript: str
    speaker_info: list[dict]
    emotion: list[dict]
    nlp: dict | None
    incongruence: list[dict]
    memory: dict | None
    retrieved_knowledge: list[dict]
    system_instructions: str
    safety_context: dict


class LlmResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    evidence_summary: str


class LlmProvider(Protocol):
    async def generate(self, context: ConversationContext) -> LlmResponse: ...
