"""NLP application service: orchestrates `NlpAnalysisStage` through the
approved, unmodified TaskRunner contract, mirroring `EmotionService`'s
pattern. Unlike emotion, NLP has no "no trained model" gate - the sentiment/
NER models are pretrained and always available once downloaded, so this
service only reports `"failed"` on a genuine model-execution error, never
`"unavailable"`.
"""
from __future__ import annotations

import uuid
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.exceptions import NotFoundError, PipelineProcessingError
from voxmind.models.message import Message
from voxmind.models.nlp_annotation import NlpAnnotation
from voxmind.repositories.message_repository import MessageRepository
from voxmind.repositories.nlp_annotation_repository import NlpAnnotationRepository
from voxmind.services.conversation_service import ConversationService
from voxmind.services.nlp.analyzer import RealNlpAnalyzer
from voxmind.services.nlp.stages import NlpAnalysisInput, NlpAnalysisOutput, NlpAnalysisStage
from voxmind.workers.task_runner import JobStatus, PipelineStage, TaskRunner, wait_for_completion


class NlpService:
    def __init__(self, session: AsyncSession, *, task_runner: TaskRunner, analyzer: RealNlpAnalyzer) -> None:
        self._session = session
        self._task_runner = task_runner
        self._analyzer = analyzer
        self._conversations = ConversationService(session)
        self._messages = MessageRepository(session)
        self._annotations = NlpAnnotationRepository(session)

    async def _get_owned_message(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID
    ) -> Message:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        message = await self._messages.get_by_id(message_id)
        if message is None or message.session_id != conversation_id:
            raise NotFoundError("Message not found.")
        return message

    async def analyze_text(self, text: str) -> NlpAnalysisOutput:
        """Runs NLP analysis on arbitrary text through TaskRunner without
        persisting - used by the RAG pipeline (context assembly, incongruence)
        where the caller owns persistence, if any."""
        handle = await self._task_runner.dispatch(
            cast(PipelineStage, NlpAnalysisStage(self._analyzer)), NlpAnalysisInput(text=text)
        )
        status = await wait_for_completion(self._task_runner, handle.job_id)
        if status.status == JobStatus.FAILED:
            raise PipelineProcessingError(status.error or "NLP analysis failed.")
        return cast(NlpAnalysisOutput, await self._task_runner.get_result(handle.job_id))

    async def process_message(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID
    ) -> NlpAnnotation:
        message = await self._get_owned_message(
            conversation_id=conversation_id, user_id=user_id, message_id=message_id
        )
        result = await self.analyze_text(message.content)
        row = await self._annotations.create(message_id=message.id, result=result.annotation)
        await self._session.commit()
        return row

    async def get_for_message(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID
    ) -> NlpAnnotation | None:
        await self._get_owned_message(
            conversation_id=conversation_id, user_id=user_id, message_id=message_id
        )
        return await self._annotations.get_for_message(message_id)
