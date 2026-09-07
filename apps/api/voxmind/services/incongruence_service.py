"""Incongruence application service: for every aligned (speaker-attributed)
turn in a message, compares real per-turn sentiment (semantic signal, run on
`AlignedTurn.text`) against that turn's real emotion prediction (vocal
signal, from `EmotionPrediction`) and persists an `IncongruenceSignal`.

Requires both an `EmotionPrediction` to exist for the turn (i.e. the emotion
pipeline has already run and a trained model is active) - if not, that turn
is skipped and reported, never fabricated. This mirrors the "unavailable
per-turn, not a blanket fake result" pattern from Phase 2/3.
"""
from __future__ import annotations

import uuid
from typing import cast

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.exceptions import NotFoundError
from voxmind.models.incongruence_signal import IncongruenceSignal
from voxmind.models.message import Message
from voxmind.repositories.emotion_prediction_repository import EmotionPredictionRepository
from voxmind.repositories.incongruence_signal_repository import IncongruenceSignalRepository
from voxmind.repositories.message_repository import MessageRepository
from voxmind.repositories.transcript_repository import TranscriptRepository
from voxmind.services.conversation_service import ConversationService
from voxmind.services.incongruence.analyzer import DeterministicIncongruenceAnalyzer
from voxmind.services.incongruence.stages import (
    IncongruenceAnalysisInput,
    IncongruenceAnalysisOutput,
    IncongruenceAnalysisStage,
)
from voxmind.services.nlp_service import NlpService
from voxmind.workers.task_runner import JobStatus, PipelineStage, TaskRunner, wait_for_completion


class IncongruenceService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        task_runner: TaskRunner,
        nlp_service: NlpService,
        analyzer: DeterministicIncongruenceAnalyzer,
    ) -> None:
        self._session = session
        self._task_runner = task_runner
        self._nlp_service = nlp_service
        self._analyzer = analyzer
        self._conversations = ConversationService(session)
        self._messages = MessageRepository(session)
        self._transcripts = TranscriptRepository(session)
        self._emotion_predictions = EmotionPredictionRepository(session)
        self._signals = IncongruenceSignalRepository(session)

    async def _get_owned_message(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID
    ) -> Message:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        message = await self._messages.get_by_id(message_id)
        if message is None or message.session_id != conversation_id:
            raise NotFoundError("Message not found.")
        return message

    async def analyze_message(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID
    ) -> list[IncongruenceSignal]:
        message = await self._get_owned_message(
            conversation_id=conversation_id, user_id=user_id, message_id=message_id
        )
        turns = await self._transcripts.get_aligned_turns(message.id)

        created: list[IncongruenceSignal] = []
        for turn in turns:
            predictions = await self._emotion_predictions.list_for_turn(turn.id)
            if not predictions:
                continue  # no vocal signal available for this turn yet - skip, don't fabricate
            vocal = predictions[0]

            nlp_result = await self._nlp_service.analyze_text(turn.text)
            semantic_signal = {
                "sentiment_label": nlp_result.annotation.sentiment_label,
                "sentiment_score": nlp_result.annotation.sentiment_score,
                "text": turn.text,
            }
            vocal_signal = {
                "predicted_label": vocal.predicted_label,
                "confidence": vocal.confidence,
                "start_ms": turn.start_ms,
                "end_ms": turn.end_ms,
                "speaker_label": turn.speaker_label,
            }

            handle = await self._task_runner.dispatch(
                cast(PipelineStage, IncongruenceAnalysisStage(self._analyzer)),
                IncongruenceAnalysisInput(semantic_signal=semantic_signal, vocal_signal=vocal_signal),
            )
            status = await wait_for_completion(self._task_runner, handle.job_id)
            if status.status == JobStatus.FAILED:
                continue
            output = cast(
                IncongruenceAnalysisOutput, await self._task_runner.get_result(handle.job_id)
            )
            row = await self._signals.create(aligned_turn_id=turn.id, result=output.signal)
            created.append(row)

        await self._session.commit()
        return created

    async def get_for_message(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID
    ) -> list[IncongruenceSignal]:
        await self._get_owned_message(
            conversation_id=conversation_id, user_id=user_id, message_id=message_id
        )
        return await self._signals.list_for_message(message_id)
