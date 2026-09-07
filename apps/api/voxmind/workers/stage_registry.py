"""Phase 9: the seam `CeleryTaskRunner` needs to turn a `(stage_name,
input_json)` pair - the only two things that safely cross the Celery/Redis
boundary - back into a real, runnable `PipelineStage` instance plus the
correctly-typed Pydantic input model, inside the worker process.

Why this exists instead of pickling the actual stage object through Redis:
every stage's constructor dependencies (`StorageBackend`, `FasterWhisperProvider`,
etc.) are cheap to reconstruct from `Settings` alone - see
`voxmind.api.deps`'s `@lru_cache` factory functions, which this module reuses
directly rather than duplicating - so there is no need to serialize a live
Python object across the broker. Doing so would also mean task payloads
could balloon with framework internals and would require the `pickle`
serializer (a real security concern for a message broker), which is exactly
what ADR 0002's "plain `def` task ... persists status/output into
`pipeline_runs`" design was written to avoid.

Each registry entry provides:
  - `input_model`: the exact Pydantic class `stage_registry` deserializes
    `PipelineRun.input_json` into before calling `stage.run(...)`.
  - `output_model`: the exact Pydantic class `CeleryTaskRunner.get_result()`
    reconstructs from `PipelineRun.output_json` - callers do real attribute
    access on the result (e.g. `result.chunks`), so a plain `dict` would
    break every existing call site; this restores the real typed object.
  - `build`: an async factory `(settings, session) -> PipelineStage | None`.
    Returns `None` only when a stage is genuinely unavailable for a reason
    that isn't a bug (no active trained emotion model, no configured TTS
    provider) - the caller must treat that as an honest "unavailable"
    outcome, never silently skip it or fabricate a result.

Every concrete stage class and every dependency factory here is imported
from its real, single, existing implementation - nothing is duplicated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Awaitable, Callable, cast

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api import deps
from voxmind.core.config import Settings
from voxmind.services.emotion.active_classifier import load_active_classifier
from voxmind.services.emotion.interfaces import Wav2Vec2EmbeddingProvider
from voxmind.services.emotion.stages import (
    AcousticFeatureExtractionStage,
    AcousticFeatureInput,
    AcousticFeatureOutput,
    EmbeddingInput,
    EmbeddingOutput,
    EmotionInferenceInput,
    EmotionInferenceOutput,
    EmotionInferenceStage,
    Wav2Vec2EmbeddingStage,
)
from voxmind.services.incongruence.stages import (
    IncongruenceAnalysisInput,
    IncongruenceAnalysisOutput,
    IncongruenceAnalysisStage,
)
from voxmind.services.knowledge.stages import (
    ChunkEmbeddingInput,
    ChunkEmbeddingOutput,
    ChunkEmbeddingStage,
    DocumentIngestionInput,
    DocumentIngestionOutput,
    DocumentIngestionStage,
)
from voxmind.services.nlp.stages import NlpAnalysisInput, NlpAnalysisOutput, NlpAnalysisStage
from voxmind.services.retrieval.stages import RerankInput, RerankOutput, RerankStage
from voxmind.services.speech.stages import (
    AlignmentInput,
    AlignmentOutput,
    AudioPreprocessingStage,
    DiarizationInput,
    DiarizationOutput,
    PreprocessInput,
    PreprocessOutput,
    SpeakerDiarizationStage,
    TranscriptAlignmentStage,
    TranscriptionInput,
    TranscriptionOutput,
    WhisperTranscriptionStage,
)
from voxmind.services.tts.stages import TtsSynthesisInput, TtsSynthesisOutput, TtsSynthesisStage
from voxmind.workers.task_runner import PipelineStage

StageFactory = Callable[[Settings, AsyncSession], Awaitable["PipelineStage | None"]]


@dataclass(frozen=True)
class StageRegistration:
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    build: StageFactory


# Every concrete stage's `run()` is narrowly typed to its own Input/Output
# pair (not the Protocol's bare `BaseModel`), which mypy correctly flags as
# a Liskov/Protocol variance mismatch - the exact same shape every existing
# TaskRunner.dispatch() call site in this codebase already has and already
# resolves with `cast(PipelineStage, ...)` (see e.g. knowledge_service.py,
# emotion_service.py). Reusing that established pattern here, not inventing
# a new one.


async def _build_audio_preprocessing(settings: Settings, _session: AsyncSession) -> PipelineStage:
    return cast(PipelineStage, AudioPreprocessingStage(deps.get_storage_backend(), settings))


async def _build_whisper_transcription(_settings: Settings, _session: AsyncSession) -> PipelineStage:
    return cast(
        PipelineStage,
        WhisperTranscriptionStage(deps.get_storage_backend(), deps.get_whisper_provider()),
    )


async def _build_speaker_diarization(_settings: Settings, _session: AsyncSession) -> PipelineStage:
    return cast(
        PipelineStage,
        SpeakerDiarizationStage(deps.get_storage_backend(), deps.get_diarization_provider()),
    )


async def _build_transcript_alignment(_settings: Settings, _session: AsyncSession) -> PipelineStage:
    return cast(PipelineStage, TranscriptAlignmentStage())


async def _build_acoustic_feature_extraction(_settings: Settings, _session: AsyncSession) -> PipelineStage:
    return cast(
        PipelineStage,
        AcousticFeatureExtractionStage(deps.get_storage_backend(), deps.get_acoustic_feature_extractor()),
    )


async def _build_wav2vec2_embedding(settings: Settings, session: AsyncSession) -> PipelineStage:
    # Mirrors EmotionService.process_message's own resolution exactly (see
    # its comment): a v2-active model needs its own fine-tuned, model-
    # version-specific provider instead of the fixed frozen singleton -
    # without this, the Celery path would silently keep using v1's
    # provider even when v2 is the active model, a real divergence from
    # the in-process path that must not exist between the two TaskRunner
    # implementations.
    loaded = await load_active_classifier(session, deps.get_storage_backend(), settings=settings)
    provider: Wav2Vec2EmbeddingProvider = deps.get_wav2vec2_provider()
    if loaded is not None:
        _classifier, _model_version, embedding_provider_override = loaded
        if embedding_provider_override is not None:
            provider = embedding_provider_override
    return cast(PipelineStage, Wav2Vec2EmbeddingStage(deps.get_storage_backend(), provider))


async def _build_emotion_inference(settings: Settings, session: AsyncSession) -> PipelineStage | None:
    loaded = await load_active_classifier(session, deps.get_storage_backend(), settings=settings)
    if loaded is None:
        return None
    classifier, _model_version, _embedding_provider_override = loaded
    return cast(PipelineStage, EmotionInferenceStage(classifier))


async def _build_nlp_analysis(_settings: Settings, _session: AsyncSession) -> PipelineStage:
    return cast(PipelineStage, NlpAnalysisStage(deps.get_nlp_analyzer()))


async def _build_incongruence_analysis(_settings: Settings, _session: AsyncSession) -> PipelineStage:
    return cast(PipelineStage, IncongruenceAnalysisStage(deps.get_incongruence_analyzer()))


async def _build_document_ingestion(settings: Settings, _session: AsyncSession) -> PipelineStage:
    return cast(PipelineStage, DocumentIngestionStage(deps.get_storage_backend(), settings))


async def _build_chunk_embedding(_settings: Settings, _session: AsyncSession) -> PipelineStage:
    return cast(PipelineStage, ChunkEmbeddingStage(deps.get_embedding_provider()))


async def _build_rerank(_settings: Settings, _session: AsyncSession) -> PipelineStage:
    return cast(PipelineStage, RerankStage(deps.get_reranker()))


async def _build_tts_synthesis(_settings: Settings, _session: AsyncSession) -> PipelineStage | None:
    provider = deps.get_tts_provider()
    if provider is None:
        return None
    return cast(PipelineStage, TtsSynthesisStage(provider))


STAGE_REGISTRY: dict[str, StageRegistration] = {
    "audio_preprocessing": StageRegistration(
        PreprocessInput, PreprocessOutput, _build_audio_preprocessing
    ),
    "whisper_transcription": StageRegistration(
        TranscriptionInput, TranscriptionOutput, _build_whisper_transcription
    ),
    "speaker_diarization": StageRegistration(
        DiarizationInput, DiarizationOutput, _build_speaker_diarization
    ),
    "transcript_alignment": StageRegistration(
        AlignmentInput, AlignmentOutput, _build_transcript_alignment
    ),
    "acoustic_feature_extraction": StageRegistration(
        AcousticFeatureInput, AcousticFeatureOutput, _build_acoustic_feature_extraction
    ),
    "wav2vec2_embedding": StageRegistration(
        EmbeddingInput, EmbeddingOutput, _build_wav2vec2_embedding
    ),
    "emotion_inference": StageRegistration(
        EmotionInferenceInput, EmotionInferenceOutput, _build_emotion_inference
    ),
    "nlp_analysis": StageRegistration(NlpAnalysisInput, NlpAnalysisOutput, _build_nlp_analysis),
    "incongruence_analysis": StageRegistration(
        IncongruenceAnalysisInput, IncongruenceAnalysisOutput, _build_incongruence_analysis
    ),
    "document_ingestion": StageRegistration(
        DocumentIngestionInput, DocumentIngestionOutput, _build_document_ingestion
    ),
    "chunk_embedding": StageRegistration(
        ChunkEmbeddingInput, ChunkEmbeddingOutput, _build_chunk_embedding
    ),
    "rerank": StageRegistration(RerankInput, RerankOutput, _build_rerank),
    "tts_synthesis": StageRegistration(TtsSynthesisInput, TtsSynthesisOutput, _build_tts_synthesis),
}

# Expensive, model-loading stages are routed to the ML queue; everything else
# (pure CPU/text/deterministic work) goes to the default queue - see
# Settings.CELERY_QUEUE_ML/CELERY_QUEUE_DEFAULT.
ML_STAGE_NAMES = frozenset(
    {
        "whisper_transcription",
        "speaker_diarization",
        "acoustic_feature_extraction",
        "wav2vec2_embedding",
        "emotion_inference",
        "nlp_analysis",
        "chunk_embedding",
        "rerank",
        "tts_synthesis",
    }
)


def queue_for_stage(stage_name: str, settings: Settings) -> str:
    return settings.CELERY_QUEUE_ML if stage_name in ML_STAGE_NAMES else settings.CELERY_QUEUE_DEFAULT


# --- Test-only: a genuinely slow stage, registered ONLY when a worker
# process is explicitly launched with VOXMIND_TEST_ENABLE_SLOW_STAGE=1 (see
# tests/integration/test_celery_task_runner.py::test_real_soft_time_limit_
# kills_a_genuinely_slow_stage). This is the only way to prove Celery's
# real signal-based soft-time-limit mechanism actually interrupts a task
# running inside a real worker subprocess - a real broker/worker round
# trip, not a simulation, needs a real stage that genuinely runs long
# enough to hit a real (test-scoped, per-message) timeout.
#
# Never active in production or in any normal test run: the env var is
# checked once, at import time, and is never set anywhere except that one
# test's own subprocess environment. This has zero effect on
# STAGE_REGISTRY, ML_STAGE_NAMES, or any other stage's behavior unless a
# worker process is deliberately started with that exact variable set.
if os.environ.get("VOXMIND_TEST_ENABLE_SLOW_STAGE") == "1":
    import asyncio

    class _SlowTestStageInput(BaseModel):
        sleep_seconds: float

    class _SlowTestStageOutput(BaseModel):
        slept_seconds: float

    class _SlowTestStage:
        name = "test_slow_stage"

        async def run(self, input: _SlowTestStageInput) -> _SlowTestStageOutput:
            await asyncio.sleep(input.sleep_seconds)
            return _SlowTestStageOutput(slept_seconds=input.sleep_seconds)

    async def _build_slow_test_stage(_settings: Settings, _session: AsyncSession) -> PipelineStage:
        return cast(PipelineStage, _SlowTestStage())

    STAGE_REGISTRY["test_slow_stage"] = StageRegistration(
        _SlowTestStageInput, _SlowTestStageOutput, _build_slow_test_stage
    )
