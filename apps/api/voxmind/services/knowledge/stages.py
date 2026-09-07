"""Pipeline stages implementing the approved, unmodified `PipelineStage`
contract - same pattern as Phase 2/3. Parsing + chunking are combined into
one stage (both cheap, CPU-only, deterministic text operations); embedding
is a separate per-chunk stage, mirroring how Phase 3 dispatches one
`Wav2Vec2EmbeddingStage` per turn rather than per message.
"""
from __future__ import annotations

from pydantic import BaseModel

from voxmind.core.config import Settings
from voxmind.services.knowledge.chunking import chunk_text
from voxmind.services.knowledge.embedding_provider import HuggingFaceEmbeddingProvider
from voxmind.services.knowledge.parsers import parse_document
from voxmind.services.storage.interfaces import StorageBackend


class DocumentIngestionInput(BaseModel):
    storage_key: str
    content_type: str
    filename: str


class DocumentIngestionOutput(BaseModel):
    chunks: list[str]


class DocumentIngestionStage:
    """Downloads the raw file from storage itself (same reasoning as Phase
    2/3's stages taking a `processed_storage_key` rather than raw audio
    bytes) - keeps stage input small and safely JSON-serializable for a
    future Celery cutover."""

    name = "document_ingestion"

    def __init__(self, storage: StorageBackend, settings: Settings) -> None:
        self._storage = storage
        self._settings = settings

    async def run(self, input: DocumentIngestionInput) -> DocumentIngestionOutput:
        raw_bytes = await self._storage.download(input.storage_key)
        text = parse_document(
            content_type=input.content_type, filename=input.filename, raw_bytes=raw_bytes
        )
        chunks = chunk_text(
            text,
            chunk_size_chars=self._settings.CHUNK_SIZE_CHARS,
            overlap_chars=self._settings.CHUNK_OVERLAP_CHARS,
        )
        return DocumentIngestionOutput(chunks=chunks)


class ChunkEmbeddingInput(BaseModel):
    text: str


class ChunkEmbeddingOutput(BaseModel):
    embedding: list[float]
    model_id: str


class ChunkEmbeddingStage:
    name = "chunk_embedding"

    def __init__(self, provider: HuggingFaceEmbeddingProvider) -> None:
        self._provider = provider

    async def run(self, input: ChunkEmbeddingInput) -> ChunkEmbeddingOutput:
        embedding = await self._provider.embed(input.text)
        return ChunkEmbeddingOutput(embedding=embedding, model_id=self._provider.model_id)
