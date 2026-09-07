from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.models.knowledge_document import KnowledgeDocument


class KnowledgeDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        session_id: uuid.UUID,
        title: str,
        original_filename: str,
        content_type: str,
        size_bytes: int,
        storage_key: str,
    ) -> KnowledgeDocument:
        row = KnowledgeDocument(
            session_id=session_id,
            title=title,
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=size_bytes,
            storage_key=storage_key,
            status="pending",
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_by_id(self, document_id: uuid.UUID) -> KnowledgeDocument | None:
        return await self._session.get(KnowledgeDocument, document_id)

    async def list_for_session(self, session_id: uuid.UUID) -> list[KnowledgeDocument]:
        stmt = (
            select(KnowledgeDocument)
            .where(KnowledgeDocument.session_id == session_id)
            .order_by(KnowledgeDocument.created_at.desc())
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def mark_processing(self, document: KnowledgeDocument) -> None:
        document.status = "processing"
        await self._session.flush()

    async def mark_completed(self, document: KnowledgeDocument, *, chunk_count: int) -> None:
        document.status = "completed"
        document.chunk_count = chunk_count
        document.processed_at = datetime.now(timezone.utc)
        await self._session.flush()

    async def mark_failed(self, document: KnowledgeDocument, *, error_message: str) -> None:
        document.status = "failed"
        document.error_message = error_message
        document.processed_at = datetime.now(timezone.utc)
        await self._session.flush()
