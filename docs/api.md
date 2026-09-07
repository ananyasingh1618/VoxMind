# API Reference (Phase 1-3)

Base path: `/api/v1`. Interactive docs at `/docs` (Swagger UI) and `/redoc` when the server is running.

Every error response has the shape `{"data": null, "error": {"code": "...", "message": "...", "detail": ...}, "request_id": "..."}`. Successful responses return the resource directly (this is a deliberate, documented deviation from the original blueprint's "every response is enveloped" note - errors are enveloped for structured handling, successes are the plain resource for simpler client typing, matching common practice at e.g. Stripe/GitHub).

## Health

`GET /api/v1/health` → `{"status": "ok", "database": "ok" | "unreachable"}`. No auth required.

## Auth

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/auth/register` | POST | none | `{email, password}` → `User`. 409 if the email is taken. |
| `/auth/login` | POST | none | `{email, password}` → `AccessTokenResponse`. Sets `voxmind_refresh` (HttpOnly) and `voxmind_csrf` cookies. 401 on bad credentials. |
| `/auth/refresh` | POST | refresh cookie + CSRF header | Rotates the refresh token, returns a new access token. 401 on invalid/expired/reused token, 403 on missing/mismatched CSRF. |
| `/auth/logout` | POST | refresh cookie + CSRF header | Revokes the current session's token family only. |
| `/auth/logout-everywhere` | POST | Bearer token + CSRF header | Revokes every active refresh token for the user, across all devices. |

## Users

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/users/me` | GET | Bearer token | Returns the current `User`. |

## Conversations

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/conversations` | POST | Bearer token | `{title?}` → `Conversation`. |
| `/conversations` | GET | Bearer token | Lists the caller's own conversations, newest first. |
| `/conversations/{id}` | GET | Bearer token | 404 if not found *or* not owned by the caller (no existence leak). |
| `/conversations/{id}` | DELETE | Bearer token | Cascades to messages and audio assets. |
| `/conversations/{id}/messages` | POST | Bearer token | `{role: "user", content}` → `Message`. `role` only accepts `"user"` - there is no way to create a fake assistant reply until Phase 4's real LLM pipeline exists. |
| `/conversations/{id}/messages` | GET | Bearer token | Lists messages in creation order. |

## Audio / speech pipeline (Phase 2)

See [docs/audio.md](audio.md) for supported formats, model configuration, and what has genuinely been verified with real models versus what's real-but-credential-gated in this environment.

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/conversations/{id}/audio` | POST | Bearer token | Multipart upload (`file` field). Content is sniffed by real magic bytes, never trusted from the client's declared type. 415 for unsupported formats, 413 over the size limit. Returns `AudioAsset` (duration/sample_rate are still null - those come from preprocessing, not upload). |
| `/conversations/{id}/audio` | GET | Bearer token | Lists original (not processed-derivative) audio assets for the conversation. |
| `/conversations/{id}/audio/{audio_id}/process` | POST | Bearer token | `{language?}` → runs the full pipeline (preprocess → Whisper → diarization → alignment) and returns `AudioProcessingJob`. Under `InProcessTaskRunner` this genuinely blocks for the duration of real model inference before responding - HTTP status is always 201 regardless of the job's own `status` field (`"completed"` or `"failed"`), matching a job/task API contract even though execution is synchronous today. On success, creates a `Message` (`role: "user"`) whose `content` is the transcript text. |
| `/conversations/{id}/audio/{audio_id}/processing` | GET | Bearer token | Lists all processing job attempts for that audio asset, newest first. |
| `/conversations/{id}/processing-jobs/{job_id}` | GET | Bearer token | Fetches a single job's status/result by id. |
| `/conversations/{id}/messages/{message_id}/transcript` | GET | Bearer token | Structured result: raw `transcript_segments` (Whisper), raw `speaker_segments` (diarization - empty array if diarization didn't run), and derived `aligned_turns` (speaker-attributed, `speaker_label: null` where genuinely unattributed). |

## Emotion intelligence (Phase 3)

See [docs/emotion.md](emotion.md) for the pipeline, model architecture, and the genuine training/evaluation results. A real trained model (`ravdess-v1`, test macro F1 = 0.690) is active, so `/emotion/process` now returns genuine predictions; `EmotionService` still refuses to serve anything if no trained+active model is registered (e.g. before any training has run), reporting `"unavailable"` rather than a fabricated prediction - see [docs/DECISIONS/0005-emotion-requires-trained-model.md](DECISIONS/0005-emotion-requires-trained-model.md).

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/conversations/{id}/messages/{message_id}/emotion/process` | POST | Bearer token | Runs emotion inference over every aligned turn in the message. Returns `EmotionProcessingJob`; `status="unavailable"` (not a fabricated result) if no trained+active model is registered - see docs/DECISIONS/0005-emotion-requires-trained-model.md. |
| `/conversations/{id}/emotion-jobs/{job_id}` | GET | Bearer token | Fetches a single emotion job's status by id. |
| `/conversations/{id}/messages/{message_id}/emotion` | GET | Bearer token | Lists persisted `EmotionPrediction`s for the message's turns (empty until processing has run). |
| `/models/{component}` | GET | Bearer token | Lists `ModelVersion` metadata for a component (e.g. `emotion_classifier`) - not conversation-scoped, since model metadata isn't private per-user data. Never exposes the artifact's storage key. |

## NLP and semantic-vocal incongruence (Phase 4)

See [docs/nlp.md](nlp.md) for the real sentiment/NER/intent/topic pipeline and the incongruence-signal computation (deliberately not framed as deception detection).

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/conversations/{id}/messages/{message_id}/nlp/process` | POST | Bearer token | Runs real sentiment/NER + deterministic intent/topic extraction on the message's text. Returns `NlpAnnotation`. |
| `/conversations/{id}/messages/{message_id}/nlp` | GET | Bearer token | Returns the latest `NlpAnnotation`, or `null` if not yet analyzed. |
| `/conversations/{id}/messages/{message_id}/incongruence/analyze` | POST | Bearer token | Computes a real per-turn incongruence signal from that turn's sentiment + emotion prediction. Turns with no emotion prediction yet are skipped, never fabricated. Returns `IncongruenceSignal[]`. |
| `/conversations/{id}/messages/{message_id}/incongruence` | GET | Bearer token | Lists persisted `IncongruenceSignal`s for the message's turns. |

## Knowledge / RAG / memory (Phase 4)

See [docs/rag.md](rag.md) for the full ingestion/retrieval/LLM/grounding pipeline and [docs/memory.md](memory.md) for conversation memory.

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/conversations/{id}/documents` | POST | Bearer token | Multipart upload (`file` field, `.txt`/`.md`/`.pdf`). 415 for unsupported formats, 413 over `MAX_DOCUMENT_UPLOAD_BYTES`. Returns `KnowledgeDocument` (`status="pending"`). |
| `/conversations/{id}/documents/{document_id}/process` | POST | Bearer token | Real parse → chunk → embed pipeline through `TaskRunner`. Returns `KnowledgeDocument` with `status`/`chunk_count`/`error_message`. |
| `/conversations/{id}/documents` | GET | Bearer token | Lists documents for the conversation, newest first. |
| `/conversations/{id}/documents/{document_id}` | GET | Bearer token | Fetches a single document's status. |
| `/conversations/{id}/memory` | GET | Bearer token | Returns the current short-term recent-turns window and the latest long-term summary (if any). |
| `/conversations/{id}/memory/summarize` | POST | Bearer token | Manually triggers the same accrual check `ask()` runs automatically; still respects `SUMMARY_TRIGGER_MESSAGE_COUNT` (returns `null` if nothing new to summarize). |
| `/conversations/{id}/ask` | POST | Bearer token | `{question}` → the full RAG flow: creates the user message, runs NLP, loads memory, retrieves knowledge (hybrid + optional rerank), assembles a token-budgeted context, calls the configured LLM provider (or reports `"unavailable"`), validates citations, computes grounding, creates the assistant message. Returns `AskResponse` (both messages, the retrieval trace, and the `Generation` record). |

## Real-time voice loop (Phase 5)

See [docs/voice.md](voice.md) for the full audio-capture → STT → analysis → RAG/LLM → TTS → playback loop, the Server-Sent-Events streaming protocol, and real interruption handling.

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/conversations/{id}/voice-turns` | POST | Bearer token | Multipart `file` upload (a recorded audio blob). Returns a real `text/event-stream` of progress events ending in `done` or `error` - see docs/voice.md for the exact event shapes. |
| `/conversations/{id}/voice-turns/{turn_id}` | GET | Bearer token | Fetches a persisted `VoiceTurn`'s status and `stage_latencies_ms`. |
| `/conversations/{id}/voice-turns/{turn_id}/audio` | GET | Bearer token | Streams the real synthesized speech audio bytes for playback. |

## Guardrails, analytics, and insights (Phase 6)

See [docs/guardrails.md](guardrails.md) for the validation/grounding/safety pipeline every LLM answer passes through, and [docs/analytics.md](analytics.md) for the dashboard and conversation-insights data model.

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/conversations/{id}/ask` | POST | Bearer token | (Extended in Phase 6) `AskResponse.guardrail` now reports the real guardrail decision (`approved`/`modified`/`blocked`), reasons, and any retrieved chunks filtered out for prompt-injection patterns - `null` only when nothing was generated at all (no LLM configured). |
| `/analytics/dashboard` | GET | Bearer token | A comprehensive real-data snapshot of the caller's own activity - latency, emotion/sentiment/intent distributions, mismatch events, retrieval usage, grounding status, model versions, failure counts, and live pipeline health. Never fabricated; genuine empty states when there's no data yet. |
| `/conversations/{id}/insights` | GET | Bearer token | A real per-message intelligence timeline (sentiment/intent/emotion/incongruence, wherever computed) plus a small set of explainable, non-diagnostic summary observations/interpretations for the conversation. |

## Evaluation (Phase 7)

See [docs/evaluation.md](evaluation.md) for the STT/emotion/retrieval/grounding/system evaluation methodology, datasets, CLI commands, and exact real results.

| Endpoint | Method | Auth | Notes |
|---|---|---|---|
| `/evaluations/dashboard` | GET | Bearer token | The most recent real `EvaluationRun` for each of stt/emotion/retrieval/grounding/system. `status="never_run"` and `latest=null` for any type that has never actually been evaluated in this environment - never a placeholder metric. Not user-scoped (an evaluation run measures a shared pipeline, not private data). |
| `/evaluations/runs?evaluation_type=...` | GET | Bearer token | Full history of real, never-overwritten runs for one evaluation type, newest first - for comparing how a metric changed across real runs over time. |

Evaluations themselves are run out-of-band via CLI (`ml/evaluation/evaluate_*.py`), not triggered over HTTP - they can take from under a second (grounding/system, pure aggregation) to several minutes (emotion, real model inference over 180 real audio clips).
