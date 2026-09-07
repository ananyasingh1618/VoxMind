"""Shared parser from a provider's raw structured-tool-call payload into
`LlmResponse` - used by every provider (Anthropic, OpenAI, the test-only
mock) so they all produce identically-shaped output regardless of how each
provider's SDK hands back the tool call payload.
"""
from __future__ import annotations

from voxmind.services.llm.interfaces import Citation, LlmResponse


def parse_structured_response(data: dict) -> LlmResponse:
    citations = [
        Citation(
            chunk_id=str(c.get("chunk_id", "")),
            document_title=str(c.get("document_title", "")),
            excerpt=str(c.get("excerpt", "")),
        )
        for c in data.get("citations", [])
    ]
    return LlmResponse(
        answer=str(data.get("answer", "")),
        citations=citations,
        confidence=float(data.get("confidence", 0.0)),
        evidence_summary=str(data.get("evidence_summary", "")),
    )
