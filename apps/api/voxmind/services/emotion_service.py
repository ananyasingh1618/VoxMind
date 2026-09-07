"""Emotion application service: orchestrates the emotion pipeline through
the approved, unmodified TaskRunner contract, exactly mirroring Phase 2's
`AudioService` pattern.

The one rule this entire module exists to enforce: **no `EmotionPrediction`
is ever produced through this service unless a real, trained, active
`ModelVersion` is registered for `component="emotion_classifier"`.** If
none exists - which is this environment's actual current state, since no
labeled dataset has been supplied (see docs/emotion.md) - processing is
reported as genuinely `"unavailable"`, the same honest pattern Phase 2
established for diarization without a Hugging Face token. This is the only
code path reachable from `api/v1/endpoints/emotion.py`; the untrained-
baseline classifier (`classifier_provider.build_untrained_baseline`) is
never imported here - it exists solely for tests. See
docs/DECISIONS/0005-emotion-requires-trained-model.md.
"""
from __future__ import annotations

import time
import uuid
from typing import cast

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.exceptions import AudioProcessingError, NotFoundError
from voxmind.models.emotion_processing_job import EmotionProcessingJob
from voxmind.models.message import Message
from voxmind.repositories.emotion_prediction_repository import EmotionPredictionRepository
from voxmind.repositories.emotion_processing_job_repository import EmotionProcessingJobRepository
from voxmind.repositories.message_repository import MessageRepository
from voxmind.repositories.transcript_repository import TranscriptRepository
from voxmind.services.conversation_service import ConversationService
from voxmind.services.emotion.active_classifier import load_active_classifier
from voxmind.services.emotion.interfaces import AcousticFeatureExtractor, Wav2Vec2EmbeddingProvider
from voxmind.services.emotion.stages import (
    AcousticFeatureExtractionStage,
    AcousticFeatureInput,
    AcousticFeatureOutput,
    EmbeddingInput,
    EmbeddingOutput,
    Wav2Vec2EmbeddingStage,
)
from voxmind.services.storage.interfaces import StorageBackend
from voxmind.workers.task_runner import JobStatus, PipelineStage, TaskRunner, wait_for_completion

logger = structlog.get_logger(__name__)


class EmotionService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        storage: StorageBackend,
        task_runner: TaskRunner,
        acoustic_extractor: AcousticFeatureExtractor,
        embedding_provider: Wav2Vec2EmbeddingProvider,
    ) -> None:
        self._session = session
        self._storage = storage
        self._task_runner = task_runner
        self._acoustic_extractor = acoustic_extractor
        self._embedding_provider = embedding_provider
        self._conversations = ConversationService(session)
        self._messages = MessageRepository(session)
        self._transcripts = TranscriptRepository(session)
        self._jobs = EmotionProcessingJobRepository(session)
        self._predictions = EmotionPredictionRepository(session)

    async def _get_owned_message(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID
    ) -> Message:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        message = await self._messages.get_by_id(message_id)
        if message is None or message.session_id != conversation_id:
            raise NotFoundError("Message not found.")
        return message

    async def _load_active_classifier(self):
        # Return type is `tuple[classifier, ModelVersion, embedding_provider_override | None] | None`
        # - see active_classifier.py's docstring. `TorchEmotionClassifier`
        # stays imported here (used by the type-narrowing `cast()` calls
        # below, and this is v1's only remaining reference in this file)
        # even though v2's classifier can also come back from this call.
        return await load_active_classifier(self._session, self._storage)

    async def process_message(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID
    ) -> EmotionProcessingJob:
        message = await self._get_owned_message(
            conversation_id=conversation_id, user_id=user_id, message_id=message_id
        )
        if message.audio_asset_id is None:
            raise NotFoundError("This message has no associated audio to analyze.")

        job = await self._jobs.create(session_id=conversation_id, message_id=message_id)
        await self._session.commit()
        log = logger.bind(
            conversation_id=str(conversation_id), message_id=str(message_id), job_id=str(job.id)
        )

        loaded = await self._load_active_classifier()
        if loaded is None:
            log.info("emotion_processing_unavailable", reason="no_trained_active_model")
            await self._jobs.mark_unavailable(
                job,
                reason=(
                    "No trained emotion model is registered yet. Train one with "
                    "ml/training/train_emotion_model.py and activate it - see docs/emotion.md."
                ),
            )
            await self._session.commit()
            return job
        classifier, model_version, embedding_provider_override = loaded
        # v1 models: embedding_provider_override is None, so the fixed,
        # shared frozen provider injected at construction is used exactly
        # as before. v2 models: a real, model-version-specific fine-tuned
        # provider is used instead for this call - see
        # active_classifier.py's and v2/inference.py's module docstrings
        # for why v2 needs this (a v2 checkpoint's Wav2Vec2 weights are
        # part of what was trained, unlike v1's fixed frozen backbone).
        embedding_provider = embedding_provider_override or self._embedding_provider

        turns = await self._transcripts.get_aligned_turns(message_id)
        processed_key = f"conversations/{conversation_id}/audio/{message.audio_asset_id}/processed.wav"

        stage_durations_ms: dict[str, int] = {"acoustic_features": 0, "embeddings": 0, "inference": 0}
        turns_processed = 0
        last_error: str | None = None

        for turn in turns:
            try:
                started = time.monotonic()
                features_handle = await self._task_runner.dispatch(
                    cast(
                        PipelineStage,
                        AcousticFeatureExtractionStage(self._storage, self._acoustic_extractor),
                    ),
                    AcousticFeatureInput(
                        processed_storage_key=processed_key, start_ms=turn.start_ms, end_ms=turn.end_ms
                    ),
                )
                features_status = await wait_for_completion(self._task_runner, features_handle.job_id)
                stage_durations_ms["acoustic_features"] += round((time.monotonic() - started) * 1000)
                if features_status.status == JobStatus.FAILED:
                    raise AudioProcessingError(
                        features_status.error or "Acoustic feature extraction failed."
                    )
                features_result = cast(
                    AcousticFeatureOutput, await self._task_runner.get_result(features_handle.job_id)
                )

                started = time.monotonic()
                embedding_handle = await self._task_runner.dispatch(
                    cast(
                        PipelineStage, Wav2Vec2EmbeddingStage(self._storage, embedding_provider)
                    ),
                    EmbeddingInput(
                        processed_storage_key=processed_key, start_ms=turn.start_ms, end_ms=turn.end_ms
                    ),
                )
                embedding_status = await wait_for_completion(self._task_runner, embedding_handle.job_id)
                stage_durations_ms["embeddings"] += round((time.monotonic() - started) * 1000)
                if embedding_status.status == JobStatus.FAILED:
                    raise AudioProcessingError(embedding_status.error or "Embedding extraction failed.")
                embedding_result = cast(
                    EmbeddingOutput, await self._task_runner.get_result(embedding_handle.job_id)
                )

                started = time.monotonic()
                prediction = await classifier.predict(
                    features_result.features, embedding_result.embedding
                )
                stage_durations_ms["inference"] += round((time.monotonic() - started) * 1000)

                await self._predictions.create(
                    aligned_turn_id=turn.id, model_version_id=model_version.id, result=prediction
                )
                turns_processed += 1
            except AudioProcessingError as exc:
                last_error = str(exc)
                log.warning("emotion_turn_failed", aligned_turn_id=str(turn.id), error=last_error)
                continue

        await self._session.commit()

        if turns_processed == 0 and turns:
            await self._jobs.mark_failed(
                job,
                error_code="emotion_inference_failed",
                error_message=last_error or "All turns failed emotion inference.",
                stage_durations_ms=stage_durations_ms,
            )
            await self._session.commit()
            return job

        await self._jobs.mark_completed(
            job,
            model_version_id=model_version.id,
            turns_processed=turns_processed,
            stage_durations_ms=stage_durations_ms,
        )
        await self._session.commit()
        log.info("emotion_processing_job_completed", turns_processed=turns_processed)
        return job

    async def get_job(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, job_id: uuid.UUID
    ) -> EmotionProcessingJob:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        job = await self._jobs.get_by_id(job_id)
        if job is None or job.session_id != conversation_id:
            raise NotFoundError("Emotion processing job not found.")
        return job

    async def get_predictions_for_message(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, message_id: uuid.UUID
    ):
        await self._get_owned_message(
            conversation_id=conversation_id, user_id=user_id, message_id=message_id
        )
        return await self._predictions.list_for_message(message_id)
