"""Deterministic test-only provider - never reachable outside `ENV=test` with
`MOCK_LLM=true` explicitly set (enforced by `Settings._validate_invariants`,
Phase 0). This exists so the RAG pipeline's context-assembly, citation-
validation, and grounding logic can be tested deterministically without a
real API key - it is never used to fabricate a "real" answer in any
reachable production or dev code path, and `build_llm_provider` never
returns it unless the caller has already set both flags.

It genuinely echoes back real chunk_ids it was actually given in
`context.retrieved_knowledge` (never invents one), so grounding-validation
tests exercise the real validation logic against real (test-fixture) data.
"""
from __future__ import annotations

from voxmind.services.llm.interfaces import Citation, ConversationContext, LlmResponse


class MockLlmProvider:
    provider_name = "mock"
    model_name = "mock-llm-v1"

    async def generate(self, context: ConversationContext) -> LlmResponse:
        citations = [
            Citation(
                chunk_id=str(chunk.get("chunk_id", "")),
                document_title=str(chunk.get("document_title", "")),
                excerpt=str(chunk.get("text", ""))[:200],
            )
            for chunk in context.retrieved_knowledge[:1]
        ]
        answer = (
            f"(mock) Based on the available context, here is a response to: {context.transcript}"
        )
        return LlmResponse(
            answer=answer,
            citations=citations,
            confidence=0.5,
            evidence_summary="Mock provider: echoed the first retrieved chunk, if any.",
        )
