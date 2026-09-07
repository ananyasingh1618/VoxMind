from __future__ import annotations

import uuid
from functools import lru_cache

from fastapi import Depends, Header
from jose import ExpiredSignatureError, JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings, get_settings
from voxmind.core.exceptions import InvalidCredentialsError, SessionExpiredError
from voxmind.core.security import decode_access_token
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.repositories.user_repository import UserRepository
from voxmind.services.emotion.acoustic_features import LibrosaAcousticFeatureExtractor
from voxmind.services.emotion.wav2vec_provider import HuggingFaceWav2Vec2Provider
from voxmind.services.incongruence.analyzer import DeterministicIncongruenceAnalyzer
from voxmind.services.knowledge.embedding_provider import HuggingFaceEmbeddingProvider
from voxmind.services.llm.factory import build_llm_provider
from voxmind.services.llm.interfaces import LlmProvider
from voxmind.services.nlp.analyzer import RealNlpAnalyzer
from voxmind.services.nlp.ner_provider import HuggingFaceNerProvider
from voxmind.services.nlp.sentiment_provider import HuggingFaceSentimentProvider
from voxmind.services.guardrails.interfaces import ModerationProvider
from voxmind.services.guardrails.moderation_provider import build_moderation_provider
from voxmind.services.retrieval.reranker import CrossEncoderReranker
from voxmind.services.speech.diarization_provider import PyannoteDiarizationProvider
from voxmind.services.speech.whisper_provider import FasterWhisperProvider
from voxmind.services.storage.factory import build_storage_backend
from voxmind.services.storage.interfaces import StorageBackend
from voxmind.services.tts.factory import build_tts_provider
from voxmind.services.tts.interfaces import TextToSpeechProvider
from voxmind.workers.task_runner import TaskRunner, build_task_runner


async def get_current_user(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise InvalidCredentialsError("Missing or malformed Authorization header.")
    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_access_token(token, settings=settings)
    except ExpiredSignatureError as exc:
        raise SessionExpiredError("Access token has expired.") from exc
    except JWTError as exc:
        raise InvalidCredentialsError("Invalid access token.") from exc

    try:
        user_id = uuid.UUID(payload["sub"])
    except (KeyError, ValueError) as exc:
        raise InvalidCredentialsError("Access token subject is invalid.") from exc

    user = await UserRepository(session).get_by_id(user_id)
    if user is None:
        raise InvalidCredentialsError("User for this token no longer exists.")
    return user


# --- Process-wide singletons ---------------------------------------------
#
# Each of these wraps either a connection pool (storage) or a lazily-loaded
# ML model (Whisper/diarization providers) that must be created once and
# reused across requests, not rebuilt per-request the way a plain FastAPI
# dependency normally would be. `lru_cache` here plays the same role it
# plays for `get_settings()` - a process-wide singleton behind a `Depends()`
# call, so tests can still override it via `app.dependency_overrides`.


@lru_cache
def get_storage_backend() -> StorageBackend:
    return build_storage_backend(get_settings())


def get_task_runner(
    settings: Settings = Depends(get_settings), session: AsyncSession = Depends(get_db)
) -> TaskRunner:
    """Not `@lru_cache`'d like the other process-wide singletons above -
    `CeleryTaskRunner` needs the current request's own `AsyncSession` to
    read/write `pipeline_runs` (never a session shared across requests or
    crossed into a worker process), so a fresh instance is built per
    request. `InProcessTaskRunner` doesn't need a session at all; it simply
    ignores these arguments."""
    if settings.TASK_RUNNER == "celery":
        return build_task_runner(settings.TASK_RUNNER, session=session, settings=settings)
    return build_task_runner(settings.TASK_RUNNER)


@lru_cache
def get_whisper_provider() -> FasterWhisperProvider:
    return FasterWhisperProvider(get_settings())


@lru_cache
def get_diarization_provider() -> PyannoteDiarizationProvider:
    return PyannoteDiarizationProvider(get_settings())


@lru_cache
def get_acoustic_feature_extractor() -> LibrosaAcousticFeatureExtractor:
    return LibrosaAcousticFeatureExtractor()


@lru_cache
def get_wav2vec2_provider() -> HuggingFaceWav2Vec2Provider:
    return HuggingFaceWav2Vec2Provider(get_settings())


@lru_cache
def get_sentiment_provider() -> HuggingFaceSentimentProvider:
    return HuggingFaceSentimentProvider(get_settings())


@lru_cache
def get_ner_provider() -> HuggingFaceNerProvider:
    return HuggingFaceNerProvider(get_settings())


@lru_cache
def get_nlp_analyzer() -> RealNlpAnalyzer:
    return RealNlpAnalyzer(get_sentiment_provider(), get_ner_provider())


@lru_cache
def get_incongruence_analyzer() -> DeterministicIncongruenceAnalyzer:
    return DeterministicIncongruenceAnalyzer()


@lru_cache
def get_embedding_provider() -> HuggingFaceEmbeddingProvider:
    return HuggingFaceEmbeddingProvider(get_settings())


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(get_settings())


@lru_cache
def get_llm_provider() -> LlmProvider | None:
    return build_llm_provider(get_settings())


@lru_cache
def get_tts_provider() -> TextToSpeechProvider | None:
    return build_tts_provider(get_settings())


@lru_cache
def get_moderation_provider() -> ModerationProvider:
    return build_moderation_provider(get_settings())
