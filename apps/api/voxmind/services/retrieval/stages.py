"""Only the pure-ML substeps of retrieval are `PipelineStage`s (embedding
generation reuses `services/knowledge/stages.py::ChunkEmbeddingStage`;
reranking is here) - vector/lexical search and fusion require direct
database access, so (matching Phase 2/3's layering: repositories are the
only layer touching a session, and stages never do) they run directly in
`RetrievalService`, not inside a stage. See that module for the full flow.
"""
from __future__ import annotations

from pydantic import BaseModel

from voxmind.services.retrieval.reranker import CrossEncoderReranker


class RerankInput(BaseModel):
    query: str
    passages: list[str]


class RerankOutput(BaseModel):
    scores: list[float]


class RerankStage:
    name = "rerank"

    def __init__(self, reranker: CrossEncoderReranker) -> None:
        self._reranker = reranker

    async def run(self, input: RerankInput) -> RerankOutput:
        scores = await self._reranker.score(input.query, input.passages)
        return RerankOutput(scores=scores)
