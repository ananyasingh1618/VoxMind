# VoxMind Architecture

This is the living architecture reference, updated as each phase lands. It reflects what is actually built today plus the approved design for what comes next - anything marked "Phase N" below is designed but not implemented.

## System diagram

```mermaid
flowchart TB
    subgraph Client["Browser"]
        UI[React SPA]
        MIC[Mic capture / Web Audio API]
    end

    subgraph API["FastAPI (apps/api)"]
        REST[REST v1 endpoints]
        AUTH[Auth: JWT + rotating refresh tokens]
    end

    subgraph Services["Service layer"]
        AUTHSVC[AuthService]
        CONVSVC[ConversationService]
        AUDIOSVC[AudioService: orchestrates speech pipeline]
        EMOTIONSVC[EmotionService: orchestrates emotion pipeline]
        RAGSVC[RagService: orchestrates the ask flow]
        VOICESVC[VoiceTurnService: orchestrates the real-time voice loop]
        GUARDSVC[GuardrailService: validation + grounding + safety -> decision]
        ANALYTICSSVC[AnalyticsService / InsightsService: real aggregate queries]
        STORAGE[StorageBackend: local / S3]
        RUNNER[TaskRunner: in_process (default) or celery (Phase 9, real)]
    end

    subgraph SpeechPipeline["speech pipeline (Phase 2 - real)"]
        PREPROC[preprocessing: ffmpeg decode/normalize]
        WHISPER[whisper_provider: faster-whisper STT]
        DIARIZE[diarization_provider: pyannote.audio]
        ALIGN[alignment: deterministic merge]
    end

    subgraph EmotionPipeline["emotion pipeline (Phase 3 - real, untrained)"]
        ACOUSTIC[acoustic_features: librosa]
        WAV2VEC[wav2vec_provider: Wav2Vec2 embeddings]
        CLASSIFIER[classifier_provider: PyTorch MLP]
    end

    subgraph MLWorkflow["offline ML workflow (ml/ - Phase 3)"]
        DATASET[datasets: RAVDESS parser + speaker-independent split]
        TRAIN[training/train_emotion_model.py]
        EVAL[evaluation/evaluate_emotion_model.py]
        MLFLOW[(MLflow tracking - optional)]
    end

    subgraph Phase4["Phase 4 (implemented)"]
        NLP[nlp: real sentiment/NER + deterministic intent/topics]
        INCONGRUENCE[incongruence: real tone/word alignment signal]
        MEMORY[memory: recent turns + LLM/extractive summaries]
        RETRIEVAL[retrieval: real hybrid RAG - vector+lexical+RRF+rerank]
        LLM[llm: context assembly + generation + grounding]
    end

    subgraph Phase5["Phase 5 (implemented)"]
        TTS[tts: real local VITS / credential-gated OpenAI TTS]
        SSE[Server-Sent-Events streaming + real interruption]
    end

    subgraph Phase6["Phase 6 (implemented)"]
        GUARDRAILS[guardrails: injection scan, output/citation validation, real safety check]
        ANALYTICS[analytics dashboard + conversation insights]
    end

    subgraph Data["Data layer"]
        PG[(PostgreSQL 16 + pgvector)]
        S3[(S3 / MinIO)]
    end

    UI <--> REST
    MIC --> VOICESVC
    REST --> AUTH --> AUTHSVC --> PG
    REST --> CONVSVC --> PG
    REST --> AUDIOSVC
    REST --> EMOTIONSVC
    REST --> RAGSVC
    REST --> VOICESVC
    VOICESVC -.-> SSE
    VOICESVC --> AUDIOSVC
    VOICESVC --> EMOTIONSVC
    VOICESVC --> RAGSVC
    VOICESVC --> TTS --> STORAGE
    VOICESVC -->|"real interruption:\nasyncio.CancelledError,\nnot polling"| PG
    AUDIOSVC --> STORAGE --> S3
    AUDIOSVC --> RUNNER
    RUNNER --> PREPROC --> WHISPER --> ALIGN
    RUNNER --> DIARIZE --> ALIGN
    ALIGN --> PG
    EMOTIONSVC -->|"only if a trained,\nactive model exists"| RUNNER
    RUNNER --> ACOUSTIC --> CLASSIFIER
    RUNNER --> WAV2VEC --> CLASSIFIER
    CLASSIFIER --> PG
    DATASET --> TRAIN --> PG
    TRAIN --> STORAGE
    TRAIN -.optional.-> MLFLOW
    TRAIN --> EVAL
    RAGSVC --> NLP --> PG
    RAGSVC --> MEMORY --> PG
    RAGSVC --> RETRIEVAL --> PG
    RETRIEVAL --> STORAGE
    RAGSVC --> LLM -->|"only if a provider\nis configured"| PG
    EMOTIONSVC -.-> INCONGRUENCE
    NLP -.-> INCONGRUENCE --> PG
    RETRIEVAL --> GUARDSVC
    LLM --> GUARDSVC --> GUARDRAILS --> PG
    REST --> ANALYTICSSVC --> ANALYTICS --> PG
```

## Repository structure

```
apps/
  api/           FastAPI backend (voxmind/ package)
    voxmind/
      core/        config, security, logging, exceptions, cookies, middleware
      db/          SQLAlchemy async engine/session, declarative base
      models/      ORM models (User, ModelVersion, RefreshToken, Conversation, Message, AudioAsset,
                   AudioProcessingJob, TranscriptSegment, SpeakerSegment, AlignedTurn,
                   EmotionProcessingJob, EmotionPrediction, NlpAnnotation, IncongruenceSignal,
                   ConversationSummary, KnowledgeDocument, KnowledgeChunk, RetrievalResult,
                   LlmGeneration, VoiceTurn, GuardrailEvaluation)
      repositories/  DB access, one per aggregate (incl. ModelVersionRepository, Phase 3;
                     KnowledgeChunkRepository's search_vector/search_lexical, Phase 4)
      schemas/     Pydantic request/response contracts
      services/
        speech/    real: preprocessing (ffmpeg), whisper_provider (faster-whisper),
                   diarization_provider (pyannote.audio), alignment (deterministic), stages
                   (PipelineStage implementations), audio_validation
        emotion/   real: acoustic_features (librosa), wav2vec_provider (Wav2Vec2), model
                   (PyTorch classifier + checkpoint), classifier_provider, stages, audio_slicing
        nlp/       real: sentiment_provider (RoBERTa), ner_provider (BERT-NER), deterministic
                   intent_classifier + topic_extractor, analyzer, stages
        incongruence/  real: valence mapping + deterministic analyzer, stages
        memory/    real: extractive_summarizer (Luhn-style); orchestration in memory_service.py
        knowledge/ real: parsers (txt/md/pdf), chunking, embedding_provider (MiniLM), stages
        retrieval/ real: hybrid (RRF fusion), reranker (cross-encoder), stages
        llm/       real: anthropic_provider, openai_provider, factory, context_assembler,
                   grounding, schema/parsing (forced structured tool-call output); mock_provider
                   is test-only (gated by MOCK_LLM+ENV=test)
        tts/       real: local_hf_provider (facebook/mms-tts-eng, VITS), openai_provider
                   (credential-gated), factory, stages
        guardrails/  real: prompt_injection (deterministic pattern scan), output_validation,
                     moderation_provider (real OpenAI moderation API, credential-gated; a real
                     deterministic keyword fallback - never skipped, unlike LLM/TTS)
        audio_service.py    orchestrates the speech pipeline through TaskRunner + persistence
        emotion_service.py  orchestrates the emotion pipeline; the ONLY path to production
                             predictions, gated on a trained+active model existing (Phase 3)
        nlp_service.py, incongruence_service.py, memory_service.py, knowledge_service.py,
        retrieval_service.py, rag_service.py   Phase 4 application services - rag_service.py's
                             generate_for_message() is shared by both POST .../ask (text) and
                             the Phase 5 voice loop (audio)
        voice_turn_service.py  Phase 5: orchestrates the full real-time voice loop (STT ->
                             analysis -> RagService.generate_for_message() -> TTS) as an async
                             generator streamed out as Server-Sent Events; the sole place real
                             interruption (asyncio.CancelledError, shielded cleanup - see
                             DECISIONS/0009) is handled
        guardrail_service.py  Phase 6: the ONLY path from a real LLM response to a persisted
                             assistant Message (and therefore to TTS) - validation, grounding,
                             safety check, decision, logging; called from generate_for_message()
        analytics_service.py, insights_service.py  Phase 6: real aggregate queries / real
                             per-conversation timeline + explainable, non-diagnostic summaries
      workers/     TaskRunner abstraction (in_process implemented; celery is a documented seam)
      api/v1/      versioned routers/endpoints (incl. voice.py's SSE streaming endpoint, Phase 5;
                   analytics.py, insights.py, Phase 6)
    alembic/       migrations
    tests/         unit/ integration/ e2e/
  web/           React + TypeScript + Vite frontend
    src/
      app/         routes, router, route guards
      components/  ui/ (design system), layout/, conversation/, voice/
      features/    auth/, conversations/, audio/, emotion/, nlp/, knowledge/, rag/, voice/,
                   analytics/, insights/ (hooks + business logic)
      stores/      zustand stores
      api/         typed fetch client
ml/              offline ML workflow (Phase 3): datasets/ (schema, RAVDESS parser, speaker-
                 independent splitting), training/ (train_emotion_model.py), evaluation/
                 (metrics.py, evaluate_emotion_model.py) - imports voxmind's real feature/
                 embedding/classifier code rather than duplicating it
docker/          docker-compose files + Dockerfiles
docs/            this file, api.md, database.md, development.md, audio.md, emotion.md, nlp.md,
                 memory.md, rag.md, voice.md, guardrails.md, analytics.md, DECISIONS/
```

## Backend layering

```
api/v1/endpoints  →  services/*  →  repositories/*  →  models (ORM)
```

Endpoints parse the request, call a service, and shape the response - they contain no business logic or SQL. Services own business logic and transaction boundaries (explicit `commit()`/`rollback()`). Repositories are the only layer that touches a SQLAlchemy session directly.

## Authentication

- **Access token**: JWT, 15-minute TTL, returned in the response body, held in memory on the client only (never localStorage, never a readable cookie), sent as `Authorization: Bearer`.
- **Refresh token**: opaque random token, stored **hashed** (SHA-256) server-side, set as an `HttpOnly` cookie scoped to `/api/v1/auth`.
- **Rotation**: every `/auth/refresh` call issues a new token and revokes the presented one, linked by a shared `family_id`.
- **Concurrency safety**: rotation takes a `SELECT ... FOR UPDATE` row lock, so two requests racing to rotate the same token are serialized by Postgres. A short grace window (`REFRESH_REUSE_GRACE_SECONDS`, default 10s) treats re-presentation of a *just-rotated* token as a benign race (returns 401, no revocation) rather than theft; reuse outside that window, or of a token revoked for any other reason, revokes the entire token family. See `apps/api/voxmind/repositories/refresh_token_repository.py` for the full reasoning.
- **CSRF**: double-submit cookie pattern on the cookie-authenticated endpoints (`/auth/refresh`, `/auth/logout*`) - a non-`HttpOnly` `voxmind_csrf` cookie must be echoed in an `X-CSRF-Token` header.
- **Logout vs. logout-everywhere**: `/auth/logout` revokes only the current session's token family; `/auth/logout-everywhere` revokes every active refresh token for the user, across every family/device.
- **Client-side**: the access token lives in a Zustand store (memory only); a silent `/auth/refresh` call on app boot re-establishes the session from the `HttpOnly` cookie, which is what makes "stay logged in across a reload" work without persisting the access token anywhere. A single-flight guard in `api/client.ts` ensures concurrent 401s trigger only one refresh call.

## Storage abstraction

`services/storage/interfaces.py` defines `StorageBackend` (`upload`/`download`/`delete`/`exists`). Two real implementations exist today: `LocalFilesystemStorage` (writes to disk, default for local dev) and `S3StorageBackend` (boto3, works against MinIO or real S3/R2 with the same code). Which one is active is the `STORAGE_BACKEND` env var - a config change, not a code change. Since Phase 2, this is genuinely wired to product-facing endpoints: `POST /conversations/{id}/audio` stores the original upload, and the preprocessing stage stores a separate canonical WAV derivative - both through this same interface, never touching the database with raw audio bytes.

## Background jobs / TaskRunner

`workers/task_runner.py` defines the `TaskRunner` contract every pipeline stage uses: `dispatch()` returns a `JobHandle` immediately, and callers poll `get_status()`/`get_result()` rather than assuming synchronous completion. `InProcessTaskRunner` (the default, used through Phase 1-8) runs stages synchronously in the caller's event loop - no Redis, no worker process. `CeleryTaskRunner` (real, Phase 9) enqueues a real Celery task through a real Redis broker; a separate worker process executes it and persists status/output to a `pipeline_runs` Postgres table (not Celery's result backend, which only ever holds a small completion marker). Selecting between them is a configuration change (`TASK_RUNNER=celery`) - zero call sites changed in `AudioService`, `EmotionService`, `KnowledgeService`, `RetrievalService`, or `VoiceTurnService`, exactly as this contract was designed in Phase 1 to allow. See [docs/celery.md](celery.md) for the full design, queue routing, reliability guarantees, and real verification evidence (a genuine Celery worker subprocess against a genuine Redis broker).

## Speech pipeline (Phase 2)

Real audio ingestion → ffmpeg preprocessing → faster-whisper transcription → pyannote.audio diarization → deterministic alignment, all as `PipelineStage` implementations dispatched through `TaskRunner`. See [docs/audio.md](audio.md) for the full detail: supported formats, model configuration, the alignment algorithm, and exactly what has been verified with real models in this environment (Whisper: yes, genuinely; diarization: implemented and unit-tested, but not executable here since it requires a gated Hugging Face model this environment has no token for - the pipeline degrades gracefully rather than faking speaker labels when that's the case).

## Emotion pipeline (Phase 3)

Real acoustic feature extraction (librosa) + real Wav2Vec2 embeddings (Hugging Face, non-gated `facebook/wav2vec2-base`) feed a real, trainable PyTorch classifier, predicting per speaker turn (`AlignedTurn`). `EmotionService` is the only path to production predictions, and it refuses to run unless a `ModelVersion` with `trained=True, is_active=True` exists for `component="emotion_classifier"` (see [docs/DECISIONS/0005-emotion-requires-trained-model.md](DECISIONS/0005-emotion-requires-trained-model.md)) - **one now does**: `ravdess-v1`, trained and evaluated on the real RAVDESS corpus (test macro F1 = 0.690, see [docs/emotion.md](emotion.md) for full metrics), verified via a correct real prediction on a held-out sample through the live API. The offline training/evaluation workflow (`ml/`) reuses these same `voxmind.services.emotion.*` classes rather than duplicating feature/model code, and logs to MLflow (a real local run exists under `apps/api/mlruns/`).

## NLP, memory, RAG, and the tone/word alignment signal (Phase 4)

Real Hugging Face sentiment/NER models plus deterministic intent/topic extraction (`services/nlp/`), a deterministic short-term/long-term memory system with real (LLM or extractive-fallback) summarization (`services/memory_service.py`), real document ingestion/chunking/embedding and hybrid (vector + lexical + RRF fusion + optional cross-encoder rerank) retrieval (`services/knowledge/`, `services/retrieval/`), and a real, credential-gated LLM provider abstraction with application-side citation validation and grounding (`services/llm/`) are all implemented - see [docs/nlp.md](nlp.md), [docs/memory.md](memory.md), and [docs/rag.md](rag.md) for the full detail, including exactly what's been verified with real models versus credential-gated.

VoxMind compares *what* is said (semantic signal) to *how* it's said (vocal signal) and surfaces divergence as an analytical cue - useful for spotting sarcasm, stress, or hesitation. This is explicitly **not** a lie detector, deception detector, or truthfulness predictor, and is never named, scored, or described as one anywhere in the codebase, API, or UI. The interface (`services/incongruence/interfaces.py`) bakes this in with a `signal_category: Literal["analytical_not_diagnostic"]` field, enforced by a database `CHECK` constraint, as a guardrail against the framing drifting under later implementation pressure.

The LLM is never asked to perform sentiment analysis, emotion detection, or retrieval itself - it consumes a `ConversationContext` object assembled by `services/llm/context_assembler.py` (a real, token-budgeted context assembler - see docs/rag.md) from upstream services' structured output, and returns a structured `LlmResponse {answer, citations, confidence, evidence_summary}` via forced tool-use/function-calling (never free-form prose parsing). `evidence_summary` is a factual summary of which retrieved facts/signals supported the answer; no field anywhere stores or exposes a model's private chain-of-thought. Citations are validated against the actually-retrieved chunk ids by the application, never trusted from the model - see docs/rag.md's grounding section.

Provider config (`LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) is real and wired to two working providers (Anthropic, OpenAI); `build_llm_provider()` returns `None` - not a fake provider - when no key is configured, and `RagService` reports that honestly as `grounding_status="unavailable"` rather than fabricating an answer. Neither key is configured in this environment, so the real network call path is implemented and unit-tested but not exercised end-to-end here (a test-only mock provider, gated behind `ENV=test` + `MOCK_LLM=true`, verifies the surrounding pipeline instead - see docs/rag.md). A Claude.ai Pro/Max or ChatGPT Plus subscription does **not** include API access - `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` must come from the respective developer consoles with their own billing.

## The real-time voice loop (Phase 5)

`VoiceTurnService` (`services/voice_turn_service.py`) wires the entire target loop - audio capture → STT → emotion/NLP/incongruence analysis → memory/RAG → LLM → TTS → playback - into one async generator, streamed to the client as real Server-Sent Events (`POST /conversations/{id}/voice-turns`) rather than a single blocking response: each event is emitted the instant the real stage it names actually finishes, never simulated client-side progress. `RagService.generate_for_message()` (extracted from `ask()` in this phase) is the shared core between a typed question and a spoken one - both a text `/ask` call and a voice turn produce an answer through identical grounding/citation/persistence logic.

Text-to-speech (`services/tts/`) mirrors the LLM provider abstraction exactly: a real local provider (`facebook/mms-tts-eng`, a non-gated VITS model - unlike the LLM providers, this one defaults *on* so the full loop is genuinely demonstrable without any credentials) and a real credential-gated cloud provider (OpenAI TTS), behind a factory that returns `None` - never a fake clip - when disabled.

Real interruption required an actual investigation, not just a design decision: an initial implementation polling `request.is_disconnected()` between stages turned out to never run in the case that mattered, because Starlette's `StreamingResponse` already cancels the request task directly via `asyncio.CancelledError` the moment it detects a disconnect. The fix - catching that cancellation once at the top level and writing `VoiceTurn.status="interrupted"` from inside a shielded `anyio.CancelScope` - is documented in full, including the live testing that found the original design wrong, in [DECISIONS/0009](DECISIONS/0009-voice-loop-cancellation.md). See [docs/voice.md](voice.md) for the complete loop, the SSE event protocol, latency instrumentation, and what's been verified live versus credential-gated.

## Guardrails, analytics, and insights (Phase 6)

`GuardrailService` (`services/guardrail_service.py`) sits between the LLM and everything downstream of it - a persisted assistant `Message`, and (through that message) TTS. It is called from exactly one place, `RagService.generate_for_message()`, so both entry points from Phase 4/5 automatically get the same protection: real prompt-injection scanning filters retrieved document chunks *before* context assembly (a chunk has no legitimate reason to contain meta-instructions to the AI; a user's own message is scanned only informationally, never blocked, since a real question can innocently contain a flagged phrase), real structural/citation validation catches an internally inconsistent response, and a real safety check (OpenAI's moderation API when `OPENAI_API_KEY` is configured, a real deterministic keyword fallback otherwise - never skipped entirely) can replace an unsafe answer with a fixed refusal. Every decision (`approved`/`modified`/`blocked`) is logged as a real, queryable `GuardrailEvaluation` row. A real correctness bug was found and fixed while wiring this in: `LlmGeneration.answer` originally stored the model's raw output even after a `blocked`/`modified` decision, meaning a client reading "the answer" could still see the unsafe/invalid text - `answer` is now overwritten with the guardrail-approved final text, with the original preserved separately (`raw_model_answer`) for audit only. See [docs/guardrails.md](guardrails.md) for the full pipeline and the live verification.

`AnalyticsService` and `InsightsService` (`services/analytics_service.py`, `services/insights_service.py`) are read-only real-aggregate-query layers over the same tables every other phase already writes to - no new pipeline stage, just real `COUNT`/`AVG`/`GROUP BY` (and, for JSONB-only fields like `VoiceTurn.stage_latencies_ms`, real Python-side aggregation over real fetched rows) scoped to the calling user's own conversations. `GET /analytics/dashboard` covers latency, emotion/sentiment/intent distributions, mismatch events, retrieval usage, grounding status, model versions, and failure counts, with genuine empty states (never fabricated numbers) when there's no data yet. `GET /conversations/{id}/insights` produces a real per-message intelligence timeline plus a small set of summary insights explicitly split into factual `observations` (each naming the real model/method that produced it) and hedged, non-diagnostic `interpretations` (every one carrying the same fixed caveat, generated only when a real pattern crosses a defined threshold) - see [docs/analytics.md](analytics.md).

## Observability

Structured logging (`structlog`) with a request-ID contextvar bound by `RequestContextMiddleware` (plain ASGI middleware as of Phase 5 - see DECISIONS/0009 for why `BaseHTTPMiddleware` was replaced) and echoed in the `X-Request-ID` response header. Every pipeline stage logs a bound-context event on completion/failure (conversation/message/job id, model version, stage name - never raw audio or credentials); `AudioProcessingJob`/`EmotionProcessingJob`/`VoiceTurn` persist real per-stage timings in `stage_durations_ms`/`stage_latencies_ms` (genuinely measured, not estimated). OpenTelemetry tracing and Prometheus metrics are still designed only (see the original blueprint decisions in `docs/DECISIONS/`), not wired in yet.

## What changed from the original blueprint

See `docs/DECISIONS/` for the itemized corrections made during architecture review (schema ordering, refresh-token concurrency, the incongruence renaming, the TaskRunner boundary, expanded auth, terminology fixes, and the LLM cost/provider strategy) - all implemented as described here.
