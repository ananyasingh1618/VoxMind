"""Analytics application service (Phase 6): every number here comes from a
real aggregate query (or a real in-memory aggregation over real rows) scoped
to the current user's own conversations - never a fabricated or estimated
value. A metric with no underlying data returns a genuine zero-sample
result (`LatencyStats(avg_ms=None, sample_count=0, ...)`, an empty
`by_label` list, etc.) - the API layer/frontend render an explicit empty
state for those rather than inventing a chart.

`retrieval_results.latency_ms`/`llm_generations.latency_ms` are simple
columns aggregated directly in SQL. STT/analysis/TTS/end-to-end latency only
exist on `VoiceTurn.stage_latencies_ms` (a JSONB blob, since only the voice
loop runs those stages) - rather than fighting Postgres JSON-path casts for
a handful of rows in a portfolio-scale app, those are aggregated in Python
over the real fetched rows, which is equally real, just not SQL-side.
"""
from __future__ import annotations

import statistics
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings
from voxmind.models.aligned_turn import AlignedTurn
from voxmind.models.audio_processing_job import AudioProcessingJob
from voxmind.models.conversation import Conversation
from voxmind.models.emotion_prediction import EmotionPrediction
from voxmind.models.emotion_processing_job import EmotionProcessingJob
from voxmind.models.guardrail_evaluation import GuardrailEvaluation
from voxmind.models.incongruence_signal import IncongruenceSignal
from voxmind.models.knowledge_chunk import KnowledgeChunk
from voxmind.models.knowledge_document import KnowledgeDocument
from voxmind.models.llm_generation import LlmGeneration
from voxmind.models.message import Message
from voxmind.models.model_version import ModelVersion
from voxmind.models.nlp_annotation import NlpAnnotation
from voxmind.models.retrieval_result import RetrievalResult
from voxmind.models.voice_turn import VoiceTurn
from voxmind.schemas.analytics import (
    AnalyticsDashboard,
    ConversationStats,
    EmotionDistribution,
    FailureCounts,
    GroundingOverview,
    IntelligenceOverview,
    LabelCount,
    LatencyOverview,
    LatencyStats,
    ModelVersionSummary,
    PipelineHealth,
    RetrievalUsage,
    SentimentTrendPoint,
)
from voxmind.services.llm.factory import build_llm_provider
from voxmind.services.tts.factory import build_tts_provider

_EMPTY_LATENCY = LatencyStats(avg_ms=None, p50_ms=None, p95_ms=None, sample_count=0)


def _latency_stats(values: list[float]) -> LatencyStats:
    if not values:
        return _EMPTY_LATENCY
    sorted_values = sorted(values)
    p50 = statistics.median(sorted_values)
    p95_index = min(len(sorted_values) - 1, round(0.95 * (len(sorted_values) - 1)))
    return LatencyStats(
        avg_ms=round(statistics.fmean(sorted_values), 1),
        p50_ms=round(p50, 1),
        p95_ms=round(sorted_values[p95_index], 1),
        sample_count=len(sorted_values),
    )


class AnalyticsService:
    def __init__(self, session: AsyncSession, *, settings: Settings) -> None:
        self._session = session
        self._settings = settings

    async def get_dashboard(self, *, user_id: uuid.UUID) -> AnalyticsDashboard:
        return AnalyticsDashboard(
            conversations=await self._conversation_stats(user_id),
            latency=await self._latency_overview(user_id),
            emotion=await self._emotion_distribution(user_id),
            intelligence=await self._intelligence_overview(user_id),
            retrieval=await self._retrieval_usage(user_id),
            grounding=await self._grounding_overview(user_id),
            model_versions=await self._model_versions(),
            failures=await self._failure_counts(user_id),
            pipeline_health=self._pipeline_health(),
        )

    async def _conversation_stats(self, user_id: uuid.UUID) -> ConversationStats:
        total = await self._scalar(
            select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
        )
        active = await self._scalar(
            select(func.count())
            .select_from(Conversation)
            .where(Conversation.user_id == user_id, Conversation.status == "active")
        )
        total_messages = await self._scalar(
            select(func.count())
            .select_from(Message)
            .join(Conversation, Message.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
        )
        return ConversationStats(
            total=total, active=active, ended=total - active, total_messages=total_messages
        )

    async def _latency_overview(self, user_id: uuid.UUID) -> LatencyOverview:
        retrieval_values = await self._scalars(
            select(RetrievalResult.latency_ms)
            .join(Conversation, RetrievalResult.session_id == Conversation.id)
            .where(Conversation.user_id == user_id, RetrievalResult.latency_ms > 0)
        )
        llm_values = await self._scalars(
            select(LlmGeneration.latency_ms)
            .join(Conversation, LlmGeneration.session_id == Conversation.id)
            .where(
                Conversation.user_id == user_id,
                # `model_name` is only ever null on the "no provider
                # configured" path (rag_service.py), where `latency_ms=0` is
                # a placeholder, not a measurement - excluding it here is
                # about whether a real provider call was attempted, not
                # whether the resulting duration happened to round to 0.
                LlmGeneration.model_name.isnot(None),
            )
        )
        voice_turn_latencies = await self._scalars(
            select(VoiceTurn.stage_latencies_ms)
            .join(Conversation, VoiceTurn.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
        )

        def _from_voice_turns(key: str) -> LatencyStats:
            values = [
                float(latencies[key])
                for latencies in voice_turn_latencies
                if latencies and latencies.get(key)
            ]
            return _latency_stats(values)

        return LatencyOverview(
            stt=_from_voice_turns("stt_ms"),
            analysis=_from_voice_turns("analysis_ms"),
            retrieval=_latency_stats([float(v) for v in retrieval_values]),
            llm=_latency_stats([float(v) for v in llm_values]),
            tts=_from_voice_turns("tts_ms"),
            end_to_end=_from_voice_turns("total_ms"),
        )

    async def _emotion_distribution(self, user_id: uuid.UUID) -> EmotionDistribution:
        stmt = (
            select(EmotionPrediction.predicted_label, func.count())
            .join(AlignedTurn, EmotionPrediction.aligned_turn_id == AlignedTurn.id)
            .join(Message, AlignedTurn.message_id == Message.id)
            .join(Conversation, Message.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
            .group_by(EmotionPrediction.predicted_label)
        )
        rows = (await self._session.execute(stmt)).all()
        by_label = [LabelCount(label=label, count=count) for label, count in rows]
        return EmotionDistribution(total_predictions=sum(c.count for c in by_label), by_label=by_label)

    async def _intelligence_overview(self, user_id: uuid.UUID) -> IntelligenceOverview:
        sentiment_stmt = (
            select(NlpAnnotation.sentiment_label, func.count())
            .join(Message, NlpAnnotation.message_id == Message.id)
            .join(Conversation, Message.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
            .group_by(NlpAnnotation.sentiment_label)
        )
        sentiment_rows = (await self._session.execute(sentiment_stmt)).all()

        trend_stmt = (
            select(
                func.date_trunc("day", NlpAnnotation.created_at).label("day"),
                NlpAnnotation.sentiment_label,
                func.count(),
            )
            .join(Message, NlpAnnotation.message_id == Message.id)
            .join(Conversation, Message.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
            .group_by("day", NlpAnnotation.sentiment_label)
            .order_by("day")
        )
        trend_rows = (await self._session.execute(trend_stmt)).all()
        trend_by_day: dict[str, dict[str, int]] = {}
        for day, label, count in trend_rows:
            key = day.date().isoformat()
            trend_by_day.setdefault(key, {"positive": 0, "neutral": 0, "negative": 0})
            if label in trend_by_day[key]:
                trend_by_day[key][label] = count

        intent_stmt = (
            select(NlpAnnotation.intent_label, func.count())
            .join(Message, NlpAnnotation.message_id == Message.id)
            .join(Conversation, Message.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
            .group_by(NlpAnnotation.intent_label)
        )
        intent_rows = (await self._session.execute(intent_stmt)).all()

        mismatch_stmt = (
            select(IncongruenceSignal.incongruence_score)
            .join(AlignedTurn, IncongruenceSignal.aligned_turn_id == AlignedTurn.id)
            .join(Message, AlignedTurn.message_id == Message.id)
            .join(Conversation, Message.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
        )
        mismatch_scores = [float(s) for s in await self._scalars(mismatch_stmt)]

        return IntelligenceOverview(
            sentiment_distribution=[LabelCount(label=label, count=count) for label, count in sentiment_rows],
            sentiment_trend=[
                SentimentTrendPoint(date=day, **counts) for day, counts in sorted(trend_by_day.items())
            ],
            intent_distribution=[LabelCount(label=label, count=count) for label, count in intent_rows],
            mismatch_event_count=len(mismatch_scores),
            mismatch_average_score=(
                round(statistics.fmean(mismatch_scores), 3) if mismatch_scores else None
            ),
            high_mismatch_event_count=sum(1 for s in mismatch_scores if s >= 0.6),
        )

    async def _retrieval_usage(self, user_id: uuid.UUID) -> RetrievalUsage:
        total_queries = await self._scalar(
            select(func.count())
            .select_from(RetrievalResult)
            .join(Conversation, RetrievalResult.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
        )
        items_lists = await self._scalars(
            select(RetrievalResult.items)
            .join(Conversation, RetrievalResult.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
        )
        selected_counts = [sum(1 for item in items if item.get("selected")) for items in items_lists]
        queries_with_results = sum(1 for c in selected_counts if c > 0)
        reranked_flags = await self._scalars(
            select(RetrievalResult.reranked)
            .join(Conversation, RetrievalResult.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
        )
        total_documents = await self._scalar(
            select(func.count())
            .select_from(KnowledgeDocument)
            .join(Conversation, KnowledgeDocument.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
        )
        total_chunks = await self._scalar(
            select(func.count())
            .select_from(KnowledgeChunk)
            .join(Conversation, KnowledgeChunk.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
        )
        return RetrievalUsage(
            total_queries=total_queries,
            queries_with_results=queries_with_results,
            average_chunks_retrieved=(
                round(statistics.fmean(selected_counts), 2) if selected_counts else None
            ),
            total_documents=total_documents,
            total_chunks=total_chunks,
            rerank_usage_rate=(
                round(sum(1 for r in reranked_flags if r) / len(reranked_flags), 2)
                if reranked_flags
                else None
            ),
        )

    async def _grounding_overview(self, user_id: uuid.UUID) -> GroundingOverview:
        stmt = (
            select(LlmGeneration.grounding_status, func.count())
            .join(Conversation, LlmGeneration.session_id == Conversation.id)
            .where(Conversation.user_id == user_id)
            .group_by(LlmGeneration.grounding_status)
        )
        rows = (await self._session.execute(stmt)).all()
        by_status = [LabelCount(label=label, count=count) for label, count in rows]
        return GroundingOverview(by_status=by_status, total_generations=sum(c.count for c in by_status))

    async def _model_versions(self) -> list[ModelVersionSummary]:
        # Model registry metadata isn't private per-user data (same
        # reasoning as GET /models/{component}, Phase 3) - shown globally.
        stmt = select(ModelVersion).order_by(ModelVersion.component, ModelVersion.created_at.desc())
        rows = (await self._session.execute(stmt)).scalars().all()
        return [
            ModelVersionSummary(
                component=row.component,
                version_tag=row.version_tag,
                is_active=row.is_active,
                trained=row.trained,
                metrics=row.metrics,
            )
            for row in rows
        ]

    async def _failure_counts(self, user_id: uuid.UUID) -> FailureCounts:
        audio_failed = await self._scalar(
            select(func.count())
            .select_from(AudioProcessingJob)
            .join(Conversation, AudioProcessingJob.session_id == Conversation.id)
            .where(Conversation.user_id == user_id, AudioProcessingJob.status == "failed")
        )
        emotion_failed = await self._scalar(
            select(func.count())
            .select_from(EmotionProcessingJob)
            .join(Conversation, EmotionProcessingJob.session_id == Conversation.id)
            .where(Conversation.user_id == user_id, EmotionProcessingJob.status == "failed")
        )
        voice_failed = await self._scalar(
            select(func.count())
            .select_from(VoiceTurn)
            .join(Conversation, VoiceTurn.session_id == Conversation.id)
            .where(Conversation.user_id == user_id, VoiceTurn.status == "failed")
        )
        voice_interrupted = await self._scalar(
            select(func.count())
            .select_from(VoiceTurn)
            .join(Conversation, VoiceTurn.session_id == Conversation.id)
            .where(Conversation.user_id == user_id, VoiceTurn.status == "interrupted")
        )
        llm_unavailable = await self._scalar(
            select(func.count())
            .select_from(LlmGeneration)
            .join(Conversation, LlmGeneration.session_id == Conversation.id)
            .where(Conversation.user_id == user_id, LlmGeneration.grounding_status == "unavailable")
        )
        guardrail_blocked = await self._scalar(
            select(func.count())
            .select_from(GuardrailEvaluation)
            .join(Conversation, GuardrailEvaluation.session_id == Conversation.id)
            .where(Conversation.user_id == user_id, GuardrailEvaluation.decision == "blocked")
        )
        guardrail_modified = await self._scalar(
            select(func.count())
            .select_from(GuardrailEvaluation)
            .join(Conversation, GuardrailEvaluation.session_id == Conversation.id)
            .where(Conversation.user_id == user_id, GuardrailEvaluation.decision == "modified")
        )
        return FailureCounts(
            audio_processing_failed=audio_failed,
            emotion_processing_failed=emotion_failed,
            voice_turns_failed=voice_failed,
            voice_turns_interrupted=voice_interrupted,
            llm_unavailable=llm_unavailable,
            guardrail_blocked=guardrail_blocked,
            guardrail_modified=guardrail_modified,
        )

    def _pipeline_health(self) -> PipelineHealth:
        settings = self._settings
        llm_provider = build_llm_provider(settings)
        tts_provider = build_tts_provider(settings)
        return PipelineHealth(
            llm_provider_configured=llm_provider is not None,
            llm_provider=settings.LLM_PROVIDER,
            tts_provider_configured=tts_provider is not None,
            tts_provider=settings.TTS_PROVIDER,
            diarization_available=bool(settings.HUGGINGFACE_TOKEN),
            moderation_provider="openai_moderation" if settings.OPENAI_API_KEY else "keyword_fallback",
        )

    async def _scalar(self, stmt: Any) -> int:
        result = await self._session.execute(stmt)
        return result.scalar_one() or 0

    async def _scalars(self, stmt: Any) -> list:
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
