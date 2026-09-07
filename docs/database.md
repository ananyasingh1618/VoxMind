# Database

PostgreSQL 16 with the `pgvector` extension enabled (enabled from Phase 1's migration; genuinely used since Phase 4 by `knowledge_chunks.embedding`, a real `vector(384)` column with an `ivfflat` cosine-distance index).

## Entities (Phase 1-3)

| Table | Purpose |
|---|---|
| `users` | Account credentials (bcrypt-hashed password) |
| `model_versions` | Model/experiment registry. Contains one real trained row (`ravdess-v1`, `component=emotion_classifier`, active) produced by `ml/training/train_emotion_model.py` against the real RAVDESS dataset (see docs/emotion.md) |
| `refresh_tokens` | Refresh-token rotation chains (`family_id`, `replaced_by_id`, `revoked_at`) |
| `conversation_sessions` | A conversation ("Conversation" in the API/UI) |
| `turns` | A message within a conversation ("Message" in the API/UI) |
| `audio_assets` | A stored audio file reference (storage key, duration, sample rate, channels, format, size). `kind` distinguishes the original upload from the derived preprocessed WAV (`source_asset_id` links the two); never holds audio bytes itself |
| `audio_processing_jobs` | One row per attempt at running the speech pipeline against an audio asset - overall `status`, plus a separately-tracked `diarization_status` (`completed`/`unavailable`/`failed`) so a missing diarization credential doesn't destroy an otherwise-successful transcript |
| `transcript_segments` | Raw, model-produced Whisper segments (own segmentation/timestamps), FK'd to the `Message` the transcript became |
| `speaker_segments` | Raw, model-produced diarization segments (own segmentation/timestamps, different boundaries than Whisper's) |
| `aligned_turns` | Derived, speaker-attributed turns from the deterministic alignment algorithm - explicitly distinct from the two raw tables above (`speaker_label` nullable: unattributed, never guessed) |
| `emotion_processing_jobs` | One row per attempt at running emotion inference over a message's aligned turns - `status="unavailable"` (not `"failed"`) when no trained+active model is registered, mirroring `audio_processing_jobs.diarization_status`'s pattern |
| `emotion_predictions` | One row per aligned turn per inference run: `predicted_label`, `confidence`, the full `probabilities` distribution, and a `model_version_id` FK (`ON DELETE RESTRICT` - a model that produced real predictions can't be deleted out from under its audit trail) |

## Entities (Phase 4)

| Table | Purpose |
|---|---|
| `nlp_annotations` | One row per message: real sentiment (`cardiffnlp/twitter-roberta-base-sentiment-latest`) + NER (`dslim/bert-base-NER`) plus deterministic intent/topic extraction. See [docs/nlp.md](nlp.md) |
| `incongruence_signals` | One row per aligned turn: a deterministic comparison of that turn's real per-turn sentiment against its real emotion prediction. `signal_category` is `CHECK`-constrained to `'analytical_not_diagnostic'` - see [docs/nlp.md](nlp.md) and [DECISIONS/0003](DECISIONS/0003-incongruence-not-deception.md) |
| `conversation_summaries` | Immutable, insert-only rolling long-term memory. `covers_message_ids` (JSONB array) records exactly which messages a summary covers; `method` honestly records `llm_generated` vs `extractive_fallback` - see [docs/memory.md](memory.md) |
| `knowledge_documents` | A user-uploaded knowledge-base document, scoped to a conversation (same ownership pattern as `audio_assets`). Raw bytes live in `StorageBackend`, never in the DB |
| `knowledge_chunks` | One real, embedded chunk of a document. `embedding` is a genuine `pgvector` `vector(384)` column (`sentence-transformers/all-MiniLM-L6-v2`, loaded via plain Transformers); lexical search matches `to_tsvector('english', content)` computed at query time (no stored tsvector column) - see [docs/rag.md](rag.md) |
| `retrieval_results` | The full fusion trace of one retrieval run - every candidate considered (not just the selected top-k), with per-source scores/ranks and the final hybrid/rerank ordering, so retrieval is inspectable and reproducible after the fact |
| `llm_generations` | One structured LLM generation attempt: `answer`, `citations`, `confidence`, `evidence_summary` - never a model's private chain-of-thought. `grounding_status` is computed by the application (citation-vs-retrieved-chunk validation), never self-reported by the model. `grounding_status="unavailable"` (not a fabricated answer) when no LLM provider is configured or the provider call fails |

## Entities (Phase 5)

| Table | Purpose |
|---|---|
| `voice_turns` | One row per pass through the real-time voice loop (audio capture → STT → analysis → RAG/LLM → TTS). `status` (`pending`/`completed`/`partial`/`failed`/`interrupted`) tracks the turn's lifecycle; `stage_latencies_ms` (JSONB) persists genuinely measured per-stage timings (`stt_ms`, `analysis_ms`, `retrieval_ms`, `llm_ms`, `tts_ms`, `total_ms`). See [docs/voice.md](voice.md) |

## Entities (Phase 6)

| Table | Purpose |
|---|---|
| `guardrail_evaluations` | One row per guardrail decision (`approved`/`modified`/`blocked`) on an `LlmGeneration` - real injection-scan/output-validation/safety-check results, filtered chunk ids, and both the original and final answer text. See [docs/guardrails.md](guardrails.md) |

## Entities (Phase 7)

| Table | Purpose |
|---|---|
| `evaluation_runs` | One row per real, executed evaluation run (`evaluation_type` in `stt`/`emotion`/`retrieval`/`grounding`/`system`). Insert-only - a new evaluation always adds a new row, never overwrites a prior one, so historical results stay inspectable. `status` distinguishes `completed`/`partial`/`failed`/`unavailable` (a metric that couldn't honestly be computed because a prerequisite dataset/ground-truth/credential is missing is `unavailable` with an explanatory `notes`, never a fabricated value). See [docs/evaluation.md](evaluation.md) |

Two existing tables were also extended in Phase 6: `retrieval_results` gained `latency_ms` (a genuinely measured retrieval duration - previously held only transiently, never persisted) and `llm_generations` gained `raw_model_answer` (the model's original, pre-guardrail output, preserved for audit only - `answer` itself is overwritten with the guardrail-approved final text, so every API/SSE consumer that reads `answer` never sees an unvalidated response). See [docs/guardrails.md](guardrails.md).

`model_versions` (extended in Phase 3) now also carries `task`, `base_model` (the exact embedding model used), `label_mapping`, `training_config`, `dataset_info`, and `artifact_storage_key` - everything needed to identify, reload, and audit a specific trained model. See docs/emotion.md.

## Entities (Phase 9)

| Table | Purpose |
|---|---|
| `pipeline_runs` | The authoritative record behind every real `CeleryTaskRunner` dispatch - one row per `PipelineStage.run()` call routed through Celery. Tracks `stage_name`, `queue`, `status` (`pending`/`running`/`completed`/`failed`), `input_json`/`output_json`, `error`, `celery_task_id`, `retry_count`, and `correlation_id` (the originating HTTP request's `request_id`, for cross-process tracing). Read by `get_status()`/`get_result()` directly - independent of whether Redis/the worker is reachable at query time. See [docs/celery.md](celery.md) |

## Naming note

The blueprint's `Conversation`/`Message` product-facing names map to `conversation_sessions`/`turns` at the table level - a naming choice carried over from the original schema design, exposed to the rest of the app as `Conversation`/`Message` ORM classes. The blueprint's `raw_text` column is implemented as `content` for clarity, since Phase 1 has no STT yet and messages are plain persisted text.

## Constraints and cascade behavior

- `users.email` - unique, indexed
- `refresh_tokens.token_hash` - unique, indexed (never stores the raw token)
- `conversation_sessions.status` - `CHECK (status IN ('active', 'ended'))`
- `turns.role` - `CHECK (role IN ('user', 'assistant'))` (enforced at the DB **and** the API only ever accepts client-supplied `role: "user"` - there is no way to create a fake "assistant" message before Phase 4's real LLM exists)
- `conversation_sessions.user_id → users.id` - `ON DELETE CASCADE`
- `turns.session_id → conversation_sessions.id` - `ON DELETE CASCADE`
- `audio_assets.session_id → conversation_sessions.id` - `ON DELETE CASCADE`
- `turns.audio_asset_id → audio_assets.id` - `ON DELETE SET NULL`
- `refresh_tokens.user_id → users.id` - `ON DELETE CASCADE`
- `refresh_tokens.replaced_by_id → refresh_tokens.id` - `ON DELETE SET NULL` (self-referential rotation chain)
- `audio_assets.kind` - `CHECK (kind IN ('original', 'processed'))`
- `audio_assets.source_asset_id → audio_assets.id` - `ON DELETE CASCADE` (self-referential: deleting the original deletes its processed derivative)
- `audio_processing_jobs.status` - `CHECK (status IN ('pending', 'running', 'completed', 'failed'))`
- `audio_processing_jobs.diarization_status` - `CHECK (diarization_status IS NULL OR diarization_status IN ('completed', 'unavailable', 'failed'))`
- `audio_processing_jobs.audio_asset_id → audio_assets.id` / `.session_id → conversation_sessions.id` - `ON DELETE CASCADE`; `.message_id → turns.id` - `ON DELETE SET NULL`
- `transcript_segments.message_id`, `speaker_segments.message_id`, `aligned_turns.message_id → turns.id` - all `ON DELETE CASCADE` (deleting a message removes its speech artifacts)
- `emotion_processing_jobs.status` - `CHECK (status IN ('pending', 'running', 'completed', 'failed', 'unavailable'))`
- `emotion_processing_jobs.message_id → turns.id` / `.session_id → conversation_sessions.id` - `ON DELETE CASCADE`; `.model_version_id → model_versions.id` - `ON DELETE SET NULL`
- `emotion_predictions.aligned_turn_id → aligned_turns.id` - `ON DELETE CASCADE`; `.model_version_id → model_versions.id` - `ON DELETE RESTRICT`
- `knowledge_documents.status` - `CHECK (status IN ('pending', 'processing', 'completed', 'failed'))`
- `knowledge_documents.session_id`, `knowledge_chunks.document_id`, `knowledge_chunks.session_id`, `conversation_summaries.session_id`, `nlp_annotations.message_id`, `incongruence_signals.aligned_turn_id`, `retrieval_results.session_id`, `retrieval_results.query_message_id`, `llm_generations.session_id`, `llm_generations.query_message_id` → their respective parents - all `ON DELETE CASCADE`
- `llm_generations.answer_message_id → turns.id` - `ON DELETE SET NULL`; `.retrieval_result_id → retrieval_results.id` - `ON DELETE SET NULL`
- `incongruence_signals.signal_category` - `CHECK (signal_category = 'analytical_not_diagnostic')`
- `llm_generations.grounding_status` - `CHECK (grounding_status IN ('grounded', 'partially_grounded', 'ungrounded', 'unavailable'))`
- `voice_turns.status` - `CHECK (status IN ('pending', 'completed', 'partial', 'failed', 'interrupted'))`
- `voice_turns.session_id → conversation_sessions.id` - `ON DELETE CASCADE`; `.user_message_id`/`.assistant_message_id → turns.id` and `.llm_generation_id → llm_generations.id` - all `ON DELETE SET NULL` (a voice turn's history stays even if the message/generation it referenced is later deleted)
- `guardrail_evaluations.decision` - `CHECK (decision IN ('approved', 'modified', 'blocked'))`
- `guardrail_evaluations.session_id → conversation_sessions.id` - `ON DELETE CASCADE`; `.llm_generation_id → llm_generations.id` - `ON DELETE SET NULL`

All of the above are exercised by real tests against a real Postgres database in `apps/api/tests/integration/test_db_constraints.py` and `test_conversations.py` (cascade delete) - not asserted by inspection alone.

## Migrations

Alembic, async-engine-aware (`apps/api/alembic/env.py` builds its connection string from application `Settings`, so there is exactly one source of truth for `DATABASE_URL`).

```bash
cd apps/api
alembic upgrade head              # apply all migrations
alembic revision --autogenerate -m "description"   # generate a new migration from model changes
alembic downgrade -1              # roll back one migration
```

The Phase 1 migration (`alembic/versions/581ec0529a9e_*.py`) creates every Phase 1 table in FK-safe order (`model_versions` and `users` first, then everything that references them) and enables the `vector` extension. The Phase 2 migration (`alembic/versions/a1562e3cec5c_*.py`) adds `audio_processing_jobs`, `transcript_segments`, `speaker_segments`, `aligned_turns`, and extends `audio_assets` with `kind`/`source_asset_id`/`original_filename`/`content_type`/`size_bytes`/`channels` - the new `kind` column ships with `server_default='original'` so the migration is safe even against a table that already has rows, and its `CHECK` constraint (not detected by Alembic's autogenerate, which doesn't diff check constraints by default) was added by hand. The Phase 3 migration (`alembic/versions/611aebcea854_*.py`) adds `emotion_processing_jobs`, `emotion_predictions`, and extends `model_versions` with `task`/`base_model`/`label_mapping`/`training_config`/`dataset_info`/`artifact_storage_key`/`created_at` - autogenerate correctly detected `emotion_processing_jobs.status`'s `CHECK` constraint this time (it does track check constraints on newly-created tables; the Phase 2 gap was specifically about *adding* a check constraint to an *existing* table via `ALTER`). The Phase 4 migration (`alembic/versions/b0882bbf0687_*.py`) adds `conversation_summaries`, `knowledge_documents`, `knowledge_chunks`, `nlp_annotations`, `retrieval_results`, `incongruence_signals`, and `llm_generations` - autogenerate correctly detected both new tables' `CHECK` constraints again, but the `ivfflat` vector index and the functional GIN full-text index were added by hand (`op.execute(...)`) since neither is expressible as a plain SQLAlchemy `Index`/autogenerate diff. The Phase 5 migration (`alembic/versions/00874856855b_*.py`) adds `voice_turns`. The Phase 6 migration (`alembic/versions/50c0eb6d9dda_*.py`) adds `guardrail_evaluations` and extends `retrieval_results` (`latency_ms`) and `llm_generations` (`raw_model_answer`) - both new `NOT NULL` columns ship with an explicit `server_default` (`'0'`/`''`) for the same reason as Phase 2's `audio_assets.kind`: the migration must be safe against tables that already have rows.

**Known, permanent autogenerate false-positive**: because those two Phase 4 indexes exist in the database but not in SQLAlchemy's model metadata (there's no `Index()` object for them - see above), every future `alembic revision --autogenerate` (and `alembic check`) will propose dropping them. This is expected and has been true since Phase 4; always delete those two `op.drop_index(...)` lines from a freshly autogenerated migration before applying it (the Phase 5 migration does this, with a comment marking exactly why) - never actually drop them.
