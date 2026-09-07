"""Application-side grounding validation - the model is never trusted to
self-report groundedness. Every citation the model returns is checked
against the chunk_ids that were *actually retrieved* for this query; any
citation referencing an id that wasn't retrieved is dropped before the
response is persisted or returned to the client, so a fabricated citation
can never reach the API surface even if a provider ignores its instructions.
"""
from __future__ import annotations

from voxmind.services.llm.interfaces import Citation

GroundingStatus = str  # "grounded" | "partially_grounded" | "ungrounded"


def validate_citations(
    citations: list[Citation], retrieved_chunk_ids: set[str]
) -> tuple[list[Citation], GroundingStatus, dict]:
    """Returns (valid_citations, grounding_status, details). Invalid
    citations are filtered out entirely, never passed through."""
    valid = [c for c in citations if c.chunk_id in retrieved_chunk_ids]
    invalid_ids = [c.chunk_id for c in citations if c.chunk_id not in retrieved_chunk_ids]

    details = {
        "citations_returned": len(citations),
        "citations_valid": len(valid),
        "invalid_citation_ids": invalid_ids,
        "retrieved_chunk_count": len(retrieved_chunk_ids),
    }

    if not retrieved_chunk_ids:
        # Nothing was retrieved for this query at all - there is no basis to
        # call any claim "grounded" in a document, regardless of what the
        # model returned.
        status: GroundingStatus = "ungrounded"
    elif citations and not invalid_ids:
        status = "grounded"
    elif valid and invalid_ids:
        status = "partially_grounded"
    else:
        status = "ungrounded"

    return valid, status, details
