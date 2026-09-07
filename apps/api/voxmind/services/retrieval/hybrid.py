"""Deterministic hybrid fusion of vector and lexical retrieval results via
Reciprocal Rank Fusion (RRF) - a standard, well-documented rank-fusion
formula (`score = sum(1 / (k + rank))` over every ranked list a candidate
appears in), not a hand-tuned weighted average. `k` (`Settings.HYBRID_RRF_K`,
default 60, the value from the original Cormack et al. RRF paper) smooths
out the impact of any single list's top rank dominating the fusion.
"""
from __future__ import annotations

import uuid

from voxmind.models.knowledge_chunk import KnowledgeChunk


class FusedCandidate:
    __slots__ = (
        "chunk",
        "vector_score",
        "vector_rank",
        "lexical_score",
        "lexical_rank",
        "hybrid_score",
    )

    def __init__(self, chunk: KnowledgeChunk) -> None:
        self.chunk = chunk
        self.vector_score: float | None = None
        self.vector_rank: int | None = None
        self.lexical_score: float | None = None
        self.lexical_rank: int | None = None
        self.hybrid_score: float = 0.0


def reciprocal_rank_fusion(
    *,
    vector_results: list[tuple[KnowledgeChunk, float]],
    lexical_results: list[tuple[KnowledgeChunk, float]],
    rrf_k: int,
) -> list[FusedCandidate]:
    candidates: dict[uuid.UUID, FusedCandidate] = {}

    for rank, (chunk, score) in enumerate(vector_results, start=1):
        candidate = candidates.setdefault(chunk.id, FusedCandidate(chunk))
        candidate.vector_score = score
        candidate.vector_rank = rank
        candidate.hybrid_score += 1.0 / (rrf_k + rank)

    for rank, (chunk, score) in enumerate(lexical_results, start=1):
        candidate = candidates.setdefault(chunk.id, FusedCandidate(chunk))
        candidate.lexical_score = score
        candidate.lexical_rank = rank
        candidate.hybrid_score += 1.0 / (rrf_k + rank)

    return sorted(candidates.values(), key=lambda c: c.hybrid_score, reverse=True)
