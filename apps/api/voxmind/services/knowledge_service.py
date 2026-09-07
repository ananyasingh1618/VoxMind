"""Knowledge ingestion application service: upload -> (separate) process,
mirroring Phase 2's `AudioService` upload/process split exactly. Real
parsing (txt/md/pdf), real deterministic chunking, and real sentence
embeddings are dispatched through the approved, unmodified TaskRunner
contract - one `DocumentIngestionStage` dispatch (parse+chunk) followed by
one `ChunkEmbeddingStage` dispatch per chunk (mirroring Phase 3's per-turn
embedding dispatch pattern).
"""
from __future__ import annotations

import uuid
from typing import cast

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings
from voxmind.core.exceptions import DocumentTooLargeError, NotFoundError, UnsupportedDocumentFormatError
from voxmind.models.knowledge_document import KnowledgeDocument
from voxmind.repositories.knowledge_chunk_repository import KnowledgeChunkRepository
from voxmind.repositories.knowledge_document_repository import KnowledgeDocumentRepository
from voxmind.services.conversation_service import ConversationService
from voxmind.services.knowledge.embedding_provider import HuggingFaceEmbeddingProvider
from voxmind.services.knowledge.parsers import SUPPORTED_CONTENT_TYPES
from voxmind.services.knowledge.stages import (
    ChunkEmbeddingInput,
    ChunkEmbeddingOutput,
    ChunkEmbeddingStage,
    DocumentIngestionInput,
    DocumentIngestionOutput,
    DocumentIngestionStage,
)
from voxmind.services.storage.interfaces import StorageBackend
from voxmind.workers.task_runner import JobStatus, PipelineStage, TaskRunner, wait_for_completion

logger = structlog.get_logger(__name__)

_SUPPORTED_EXTENSIONS = {"txt", "md", "markdown", "pdf"}


def _is_supported(*, content_type: str, filename: str) -> bool:
    if content_type in SUPPORTED_CONTENT_TYPES:
        return True
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in _SUPPORTED_EXTENSIONS


class KnowledgeService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        settings: Settings,
        storage: StorageBackend,
        task_runner: TaskRunner,
        embedding_provider: HuggingFaceEmbeddingProvider,
    ) -> None:
        self._session = session
        self._settings = settings
        self._storage = storage
        self._task_runner = task_runner
        self._embedding_provider = embedding_provider
        self._conversations = ConversationService(session)
        self._documents = KnowledgeDocumentRepository(session)
        self._chunks = KnowledgeChunkRepository(session)

    async def _get_owned_document(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> KnowledgeDocument:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        document = await self._documents.get_by_id(document_id)
        if document is None or document.session_id != conversation_id:
            raise NotFoundError("Knowledge document not found.")
        return document

    async def upload(
        self,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        filename: str,
        content_type: str,
        raw_bytes: bytes,
    ) -> KnowledgeDocument:
        if len(raw_bytes) > self._settings.MAX_DOCUMENT_UPLOAD_BYTES:
            raise DocumentTooLargeError(
                f"Document exceeds the {self._settings.MAX_DOCUMENT_UPLOAD_BYTES} byte limit."
            )
        if not _is_supported(content_type=content_type, filename=filename):
            raise UnsupportedDocumentFormatError(
                f"Unsupported document type: {content_type!r} ({filename!r}). "
                "Supported: .txt, .md, .pdf."
            )
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)

        document = await self._documents.create(
            session_id=conversation_id,
            title=filename,
            original_filename=filename,
            content_type=content_type,
            size_bytes=len(raw_bytes),
            storage_key="",
        )
        storage_key = f"conversations/{conversation_id}/documents/{document.id}/{filename}"
        await self._storage.upload(storage_key, raw_bytes, content_type=content_type)
        document.storage_key = storage_key
        await self._session.commit()
        return document

    async def process(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> KnowledgeDocument:
        document = await self._get_owned_document(
            conversation_id=conversation_id, user_id=user_id, document_id=document_id
        )
        await self._documents.mark_processing(document)
        await self._session.commit()
        log = logger.bind(conversation_id=str(conversation_id), document_id=str(document_id))

        ingestion_handle = await self._task_runner.dispatch(
            cast(PipelineStage, DocumentIngestionStage(self._storage, self._settings)),
            DocumentIngestionInput(
                storage_key=document.storage_key,
                content_type=document.content_type,
                filename=document.original_filename,
            ),
        )
        ingestion_status = await wait_for_completion(self._task_runner, ingestion_handle.job_id)
        if ingestion_status.status == JobStatus.FAILED:
            await self._documents.mark_failed(
                document, error_message=ingestion_status.error or "Document parsing failed."
            )
            await self._session.commit()
            log.warning("document_ingestion_failed", error=ingestion_status.error)
            return document
        ingestion_result = cast(
            DocumentIngestionOutput, await self._task_runner.get_result(ingestion_handle.job_id)
        )

        if not ingestion_result.chunks:
            await self._documents.mark_failed(
                document, error_message="Document produced no extractable text/chunks."
            )
            await self._session.commit()
            return document

        chunk_count = 0
        last_error: str | None = None
        for index, chunk_text_value in enumerate(ingestion_result.chunks):
            embed_handle = await self._task_runner.dispatch(
                cast(PipelineStage, ChunkEmbeddingStage(self._embedding_provider)),
                ChunkEmbeddingInput(text=chunk_text_value),
            )
            embed_status = await wait_for_completion(self._task_runner, embed_handle.job_id)
            if embed_status.status == JobStatus.FAILED:
                last_error = embed_status.error
                log.warning("chunk_embedding_failed", chunk_index=index, error=last_error)
                continue
            embed_result = cast(
                ChunkEmbeddingOutput, await self._task_runner.get_result(embed_handle.job_id)
            )
            await self._chunks.create(
                document_id=document.id,
                session_id=conversation_id,
                chunk_index=index,
                document_title=document.title,
                content=chunk_text_value,
                embedding=embed_result.embedding,
                embedding_model=embed_result.model_id,
            )
            chunk_count += 1

        if chunk_count == 0:
            await self._documents.mark_failed(
                document, error_message=last_error or "All chunks failed embedding."
            )
        else:
            await self._documents.mark_completed(document, chunk_count=chunk_count)
        await self._session.commit()
        log.info("document_processing_completed", chunk_count=chunk_count)
        return document

    async def list_for_conversation(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> list[KnowledgeDocument]:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        return await self._documents.list_for_session(conversation_id)

    async def get(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, document_id: uuid.UUID
    ) -> KnowledgeDocument:
        return await self._get_owned_document(
            conversation_id=conversation_id, user_id=user_id, document_id=document_id
        )
