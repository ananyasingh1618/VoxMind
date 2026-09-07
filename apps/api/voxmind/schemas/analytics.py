"""Analytics dashboard response shapes (Phase 6). Every field is populated
from a real aggregate query over the current user's own data - see
services/analytics_service.py. An empty list/zero count is a genuine
"no data yet" result, never a fabricated placeholder - the frontend renders
an explicit empty state for those rather than inventing a chart.
"""
from __future__ import annotations

from pydantic import BaseModel


class ConversationStats(BaseModel):
    total: int
    active: int
    ended: int
    total_messages: int


class LatencyStats(BaseModel):
    """`sample_count=0` means no real measurements exist yet - `avg_ms` is
    `None` in that case, never a fabricated number."""

    avg_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    sample_count: int


class LatencyOverview(BaseModel):
    stt: LatencyStats
    analysis: LatencyStats
    retrieval: LatencyStats
    llm: LatencyStats
    tts: LatencyStats
    end_to_end: LatencyStats


class LabelCount(BaseModel):
    label: str
    count: int


class EmotionDistribution(BaseModel):
    total_predictions: int
    by_label: list[LabelCount]


class SentimentTrendPoint(BaseModel):
    date: str
    positive: int
    neutral: int
    negative: int


class IntelligenceOverview(BaseModel):
    sentiment_distribution: list[LabelCount]
    sentiment_trend: list[SentimentTrendPoint]
    intent_distribution: list[LabelCount]
    mismatch_event_count: int
    mismatch_average_score: float | None
    high_mismatch_event_count: int


class RetrievalUsage(BaseModel):
    total_queries: int
    queries_with_results: int
    average_chunks_retrieved: float | None
    total_documents: int
    total_chunks: int
    rerank_usage_rate: float | None


class GroundingOverview(BaseModel):
    by_status: list[LabelCount]
    total_generations: int


class ModelVersionSummary(BaseModel):
    component: str
    version_tag: str
    is_active: bool
    trained: bool
    metrics: dict | None


class FailureCounts(BaseModel):
    audio_processing_failed: int
    emotion_processing_failed: int
    voice_turns_failed: int
    voice_turns_interrupted: int
    llm_unavailable: int
    guardrail_blocked: int
    guardrail_modified: int


class PipelineHealth(BaseModel):
    llm_provider_configured: bool
    llm_provider: str
    tts_provider_configured: bool
    tts_provider: str
    diarization_available: bool
    moderation_provider: str


class AnalyticsDashboard(BaseModel):
    conversations: ConversationStats
    latency: LatencyOverview
    emotion: EmotionDistribution
    intelligence: IntelligenceOverview
    retrieval: RetrievalUsage
    grounding: GroundingOverview
    model_versions: list[ModelVersionSummary]
    failures: FailureCounts
    pipeline_health: PipelineHealth
