"""Central application configuration.

Every environment-dependent value is read exactly once, here, via pydantic-settings.
No other module reads `os.environ` directly — this is the single seam that makes
providers (DB, storage, LLM, task runner) swappable via env vars alone.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# apps/api/ - anchors STORAGE_LOCAL_ROOT's default below so it resolves to
# the same absolute directory regardless of the process's current working
# directory. A CWD-relative default caused a genuine bug during Phase 3
# training: `uvicorn` is documented to run from apps/api, but
# `python -m ml.training.train_emotion_model` is documented to run from the
# repo root - with a relative default, the two processes silently wrote/read
# checkpoints under two different directories, and a trained model became
# invisible to the server that was supposed to serve it.
_API_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Application ---
    ENV: Literal["dev", "test", "staging", "prod"] = "dev"
    APP_NAME: str = "voxmind-api"
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["console", "json"] = "console"

    # --- CORS ---
    CORS_ALLOWED_ORIGINS: list[str] = ["http://localhost:5173"]

    # --- Database ---
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://voxmind:voxmind@localhost:5432/voxmind",
        description="Async SQLAlchemy connection string (asyncpg driver).",
    )

    # --- Auth ---
    JWT_SECRET_KEY: str = Field(..., description="Required. Generate with `openssl rand -hex 32`.")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_TTL_MINUTES: int = 15
    REFRESH_TOKEN_TTL_DAYS: int = 7
    REFRESH_COOKIE_NAME: str = "voxmind_refresh"
    CSRF_COOKIE_NAME: str = "voxmind_csrf"
    # Local dev over plain HTTP cannot set Secure+SameSite=None cookies (browsers drop them).
    # This flag documents that relaxation explicitly rather than silently weakening prod config.
    COOKIE_SECURE: bool = True
    COOKIE_SAMESITE: Literal["strict", "lax", "none"] = "none"
    WS_TICKET_TTL_SECONDS: int = 45
    # Window during which a just-rotated (not explicitly revoked) refresh token
    # may be re-presented without triggering full family revocation — absorbs
    # legitimate concurrent refresh races (double network retry, duplicate tab
    # request) without treating them as token theft. See refresh_token_repository.
    REFRESH_REUSE_GRACE_SECONDS: int = 10

    # --- Object storage ---
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    # Absolute by default (anchored to apps/api/, not the process's CWD) -
    # see the _API_ROOT comment above for why a relative default was a real
    # bug. An explicit override (e.g. in .env) should also be an absolute
    # path for the same reason.
    STORAGE_LOCAL_ROOT: str = Field(default_factory=lambda: str(_API_ROOT / ".data" / "storage"))
    S3_ENDPOINT_URL: str | None = None
    S3_BUCKET: str = "voxmind-audio"
    S3_ACCESS_KEY: str | None = None
    S3_SECRET_KEY: str | None = None
    S3_REGION: str = "us-east-1"

    # --- Background jobs ---
    TASK_RUNNER: Literal["in_process", "celery"] = "in_process"
    # Broker + result-backend for CeleryTaskRunner (Phase 9). The result
    # backend only ever stores Celery's own small completion marker for a
    # task, never real pipeline output - see workers/tasks.py and ADR 0002.
    # Separate logical DB indices (`/0` vs `/1`) keep broker traffic and
    # result-backend traffic inspectable independently with `redis-cli`.
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_RESULT_BACKEND_URL: str = "redis://localhost:6379/1"
    # A third logical DB index for rate-limit counters (see core/rate_limit.py)
    # - the same broker-traffic/result-backend separation convention as the
    # two URLs above, extended to a third, unrelated traffic type, not a new
    # database or a new piece of infrastructure.
    REDIS_RATE_LIMIT_URL: str = "redis://localhost:6379/2"
    # Queue names: expensive, model-loading work (STT/diarization/emotion/
    # embeddings/reranking/TTS) is routed separately from cheap, pure-CPU or
    # I/O-bound work (alignment, NLP, incongruence, document parsing) so an
    # operator can scale/isolate worker pools per workload class without any
    # code change - see docs/celery.md.
    CELERY_QUEUE_ML: str = "voxmind.ml"
    CELERY_QUEUE_DEFAULT: str = "voxmind.default"
    CELERY_TASK_TIME_LIMIT_SECONDS: int = 300
    CELERY_TASK_SOFT_TIME_LIMIT_SECONDS: int = 270
    CELERY_TASK_MAX_RETRIES: int = 3
    CELERY_TASK_RETRY_BACKOFF_SECONDS: int = 5
    # How long a `pipeline_runs` row may sit at status="running" before
    # workers/reconciliation.py treats it as abandoned rather than
    # genuinely in-progress - see that module's docstring and
    # docs/DECISIONS/0017 for the full derivation. Must stay comfortably
    # above broker_transport_options["visibility_timeout"]
    # (2 * CELERY_TASK_TIME_LIMIT_SECONDS = 600s by default) plus one full
    # redelivered execution attempt (another CELERY_TASK_TIME_LIMIT_SECONDS)
    # - a row genuinely still cycling through Celery's own redelivery has
    # its started_at refreshed on every re-execution attempt (workers/
    # tasks.py always calls mark_running(), even on redelivery), so it
    # never looks stale to this threshold; only a row nothing has touched
    # in this entire window is a real reconciliation candidate.
    CELERY_RECONCILIATION_STALE_THRESHOLD_SECONDS: int = 1200
    # How often Celery Beat fires the reconciliation check itself
    # (workers/tasks.py::reconcile_stale_pipeline_runs_task) - independent
    # of, and deliberately much shorter than, the stale threshold above.
    # 300s (5 minutes) means a genuinely abandoned row is caught within a
    # few minutes of crossing the 1200s threshold, not indefinitely, while
    # staying a lightweight, infrequent check (a single indexed query,
    # normally matching zero rows) - not a tight polling loop.
    CELERY_RECONCILIATION_INTERVAL_SECONDS: int = 300

    # --- Rate limiting (core/rate_limit.py) ---
    # Protects this API from abuse/runaway expensive requests - not a
    # rate limiter for Groq or Hugging Face, and never applied to internal
    # Celery task execution. Redis-backed fixed-window counters (see the
    # module for why that specific algorithm), one shared window size, a
    # distinct request count per endpoint category below.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    # Fail-open (allow the request through, log loudly) by default so a
    # transient Redis blip degrades to "temporarily unprotected" rather than
    # "API down" - the right default for an abuse guard, not a security
    # boundary like auth. Never silent either way: every failure is logged.
    # Flip to fail-closed for a deployment that would rather reject traffic
    # than run even briefly unprotected.
    RATE_LIMIT_FAIL_OPEN: bool = True
    # No reverse proxy sits in front of this API in any deployment this
    # project currently ships (see docker-compose.full.yml - the API
    # container's port is published directly) - trusting a client-supplied
    # X-Forwarded-For without one would let any caller spoof their own
    # rate-limit identity. Only flip this once a real, controlling proxy
    # that sets this header itself is guaranteed to sit in front.
    RATE_LIMIT_TRUST_PROXY_HEADERS: bool = False
    # Category A - unauthenticated auth endpoints (register/login/refresh),
    # keyed by client IP - the classic brute-force target.
    RATE_LIMIT_AUTH_PER_WINDOW: int = 10
    # Category B - endpoints that trigger real ML inference or a real,
    # possibly-billed external LLM call (audio upload/processing,
    # voice-turns, emotion inference, /ask) - keyed by user.
    RATE_LIMIT_EXPENSIVE_PER_WINDOW: int = 15
    # Category C - knowledge/document ingestion (parsing, chunking,
    # embedding) - keyed by user.
    RATE_LIMIT_KNOWLEDGE_PER_WINDOW: int = 20
    # Category D - ordinary authenticated API traffic (everything else
    # except /health) - keyed by user.
    RATE_LIMIT_DEFAULT_PER_WINDOW: int = 120

    # --- LLM provider (config seam only — no LLM calls happen until Phase 4) ---
    LLM_PROVIDER: Literal["anthropic", "openai", "local_dev"] = "local_dev"
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    # None = OpenAI's own default endpoint (openai-python's built-in default,
    # never hardcoded here). Set to point OpenAiLlmProvider at any other
    # OpenAI-Chat-Completions-compatible endpoint (e.g. Groq's
    # https://api.groq.com/openai/v1) without a new provider class - the
    # request/response shape (including forced tool_choice) is the same API
    # this provider already speaks. Not secret - the API key is what's
    # sensitive, not which compatible host it's sent to.
    OPENAI_BASE_URL: str | None = None
    LLM_LOCAL_DEV_MODEL: str | None = None
    LLM_MAX_OUTPUT_TOKENS: int = 1024
    MOCK_LLM: bool = False

    # --- Audio ingestion ---
    # Extensions this deployment claims to support end-to-end. "webm" is
    # accepted at ingestion (browsers commonly record to it) but is not
    # guaranteed decodable by every ffmpeg build - see docs/audio.md.
    SUPPORTED_AUDIO_EXTENSIONS: list[str] = ["wav", "mp3", "m4a", "ogg", "webm"]
    MAX_AUDIO_UPLOAD_BYTES: int = 25 * 1024 * 1024  # 25 MB
    MAX_AUDIO_DURATION_SECONDS: int = 600  # 10 minutes

    # --- Speech-to-text (faster-whisper) ---
    WHISPER_MODEL_SIZE: str = "tiny"
    WHISPER_DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_LANGUAGE: str | None = None  # None = auto-detect
    WHISPER_BEAM_SIZE: int = 5

    # --- Speaker diarization (pyannote.audio) ---
    # pyannote/speaker-diarization-3.1 (and its segmentation-3.0 dependency)
    # are gated models on Hugging Face: a free HF account must accept both
    # models' license terms, and HUGGINGFACE_TOKEN must be a read-scoped
    # access token for that account. See docs/audio.md for exact steps.
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"
    HUGGINGFACE_TOKEN: str | None = None

    # --- Emotion intelligence (Phase 3) ---
    # facebook/wav2vec2-base is NOT gated - freely downloadable, unlike the
    # diarization model above. See docs/emotion.md.
    WAV2VEC2_MODEL: str = "facebook/wav2vec2-base"
    WAV2VEC2_DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"
    WAV2VEC2_POOLING: Literal["mean"] = "mean"  # masked mean pooling over time - see docs/emotion.md
    EMOTION_MODEL_ARTIFACT_ROOT: str = "models/emotion"  # storage-key prefix, not a filesystem path

    # --- Emotion2Vec+ production candidate ---
    # emotion2vec/emotion2vec_plus_base via FunASR - not gated, MIT-licensed
    # framework (model card license is indirect - see docs/emotion.md). Not
    # loadable via plain transformers - requires the `funasr` package.
    EMOTION2VEC_MODEL: str = "emotion2vec/emotion2vec_plus_base"
    EMOTION2VEC_HUB: Literal["hf", "ms"] = "hf"  # "hf" = huggingface (used outside China)
    EMOTION2VEC_DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"

    # --- NLP (Phase 4) ---
    # cardiffnlp's roberta is trained for 3-class (negative/neutral/positive)
    # sentiment, not gated, freely downloadable - same "real, no credentials"
    # bar as facebook/wav2vec2-base in Phase 3.
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    NER_MODEL: str = "dslim/bert-base-NER"
    NLP_DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"

    # --- Knowledge ingestion / embeddings (Phase 4) ---
    # A standard sentence-embedding checkpoint, not gated. Loaded via plain
    # transformers (AutoModel + mean pooling), the same pattern as
    # HuggingFaceWav2Vec2Provider - the `sentence-transformers` package
    # itself is never required.
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    EMBEDDING_DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"
    CHUNK_SIZE_CHARS: int = 1000
    CHUNK_OVERLAP_CHARS: int = 150
    MAX_DOCUMENT_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB

    # --- Retrieval (Phase 4) ---
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_CANDIDATE_K: int = 20  # candidates pulled from each of vector/lexical before fusion
    HYBRID_RRF_K: int = 60  # standard reciprocal-rank-fusion smoothing constant
    RERANK_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Memory (Phase 4) ---
    MEMORY_RECENT_TURNS: int = 8  # short-term window handed to the context assembler verbatim
    SUMMARY_TRIGGER_MESSAGE_COUNT: int = 12  # summarize once this many un-summarized messages accrue

    # --- Context assembly / LLM generation (Phase 4) ---
    # No real tokenizer is bundled for every provider (Anthropic ships none
    # offline); this is a documented, conservative characters-per-token
    # approximation used only for budgeting, never for billing or exact
    # truncation guarantees.
    CONTEXT_TOKEN_BUDGET: int = 3000
    CHARS_PER_TOKEN_ESTIMATE: int = 4
    LLM_MODEL: str = "claude-sonnet-5"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # --- TTS / voice loop (Phase 5) ---
    # Unlike LLM_PROVIDER (default "local_dev" = off), TTS defaults to a
    # real, always-available local model - facebook/mms-tts-eng is not
    # gated and needs no API key, so the full voice loop is genuinely
    # demonstrable end-to-end without any credentials. "openai" is the real
    # cloud alternative (credential-gated, same unavailable-not-fabricated
    # pattern as the LLM providers); "disabled" turns TTS off entirely.
    TTS_PROVIDER: Literal["local_hf", "openai", "disabled"] = "local_hf"
    TTS_LOCAL_MODEL: str = "facebook/mms-tts-eng"
    TTS_DEVICE: Literal["cpu", "cuda", "auto"] = "cpu"
    OPENAI_TTS_MODEL: str = "tts-1"
    OPENAI_TTS_VOICE: str = "alloy"
    VOICE_TURN_MAX_ANSWER_CHARS: int = 2000  # truncate before TTS - real synthesis time scales with length

    @model_validator(mode="after")
    def _validate_invariants(self) -> "Settings":
        if self.MOCK_LLM and self.ENV != "test":
            raise ValueError("MOCK_LLM=true is only permitted when ENV=test.")
        if self.ENV == "prod" and not self.COOKIE_SECURE:
            raise ValueError("COOKIE_SECURE must be true outside local development.")
        if self.STORAGE_BACKEND == "s3" and not (self.S3_ACCESS_KEY and self.S3_SECRET_KEY):
            raise ValueError("S3_ACCESS_KEY and S3_SECRET_KEY are required when STORAGE_BACKEND=s3.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
