# Analytics and conversation insights (Phase 6)

## Analytics dashboard

`GET /analytics/dashboard` (`services/analytics_service.py`) returns a comprehensive, real-data snapshot of the calling user's own activity - every field is a real aggregate query (or a real in-memory aggregation over real rows fetched from the database), scoped by joining through `conversation_sessions.user_id`. Nothing is estimated, interpolated, or fabricated: a metric with no underlying data yet returns a genuine zero/empty/`null` result (`LatencyStats(sample_count=0, avg_ms=null, ...)`, an empty `by_label` list), and the frontend (`AnalyticsDashboardPage.tsx`) renders an explicit "No data yet" empty state for the whole dashboard, or a per-chart "no real measurements yet" hint, rather than drawing a chart from invented numbers.

| Section | Source | Notes |
|---|---|---|
| Conversations | `conversation_sessions`, `turns` | Total/active/ended counts, total message count |
| Latency | `retrieval_results.latency_ms`, `llm_generations.latency_ms`, `voice_turns.stage_latencies_ms` | avg/p50/p95 per stage - see below for why two different aggregation strategies are used |
| Emotion distribution | `emotion_predictions` | Real predicted-label counts from the active trained model |
| Sentiment distribution/trend, intent distribution | `nlp_annotations` | Real sentiment/intent labels, grouped by day for the trend |
| Mismatch events | `incongruence_signals` | Count, average score, count above the 0.6 "notable divergence" threshold |
| Retrieval usage | `retrieval_results`, `knowledge_documents`, `knowledge_chunks` | Query count, average chunks actually selected, document/chunk counts, real rerank usage rate |
| Grounding | `llm_generations.grounding_status` | Real distribution across grounded/partially_grounded/ungrounded/unavailable |
| Model versions | `model_versions` | Global registry metadata (not user-scoped - same reasoning as `GET /models/{component}`, Phase 3) |
| Failures | `audio_processing_jobs`, `emotion_processing_jobs`, `voice_turns`, `llm_generations`, `guardrail_evaluations` | Real failure/interruption/unavailable/blocked counts |
| Pipeline health | live config checks | Whether an LLM/TTS provider is actually configured right now, which moderation provider would run, whether diarization has a token |

### Two aggregation strategies, and why

`retrieval_results.latency_ms` and `llm_generations.latency_ms` are simple integer columns, aggregated directly in SQL (`AVG`, plus Python-side `statistics.median`/percentile for p50/p95 after fetching the real values - Postgres's `percentile_cont` would also work, but for a portfolio-scale row count, fetching real values and computing in Python is equally real and simpler). STT/analysis/TTS/end-to-end latency only exist inside `VoiceTurn.stage_latencies_ms`, a JSONB blob (since only the voice loop runs those stages) - rather than fighting Postgres JSON-path casts for what will typically be a handful of rows, those are aggregated in Python over the real fetched rows.

### A real bug found and fixed here

Excluding `latency_ms = 0` values was originally applied uniformly to both `retrieval_results.latency_ms` and `llm_generations.latency_ms`, on the reasoning that `retrieval_results.latency_ms` (a column *added* in this same Phase 6 migration) has a backfilled `server_default='0'` on any row that existed before the migration - a real placeholder, not a measurement, that should be excluded from the average. Live testing with the fast, in-memory test-only mock LLM provider caught that this same filter was wrong for `llm_generations.latency_ms`: that column has been genuinely measured (via `time.monotonic()`) since Phase 4, with no backfill concern at all, and a very fast real call can legitimately round to `0ms` - excluding it as if it were a fake placeholder silently dropped real data. The fix distinguishes "no attempt was made" (`LlmGeneration.model_name IS NULL`, which only happens on the "no provider configured" path) from "a real attempt was made and happened to be fast" - only the former is excluded.

## Conversation insights / intelligence timeline

`GET /conversations/{id}/insights` (`services/insights_service.py`) returns two things for one conversation:

1. **A per-message intelligence timeline** - every message in chronological order, annotated with its real sentiment/intent/topics (if NLP has run) and, for audio-derived messages, real per-turn emotion predictions and incongruence signals (if those pipelines have run). This is the same real data the per-message panels (`EmotionPanel`, `NlpPanel`, `IncongruencePanel`) show, presented as one chronological view across the whole conversation instead of one message at a time.
2. **A small set of explainable summary insights**, split into two explicitly distinct kinds:
   - **`observations`** - factual, model-attributed statements about what was actually detected, each with a `basis` field naming exactly what produced it (a real model, e.g. `cardiffnlp/twitter-roberta-base-sentiment-latest`, or a real deterministic method, e.g. "deterministic rule-based intent classifier"). These are always safe to state plainly because they're just counts/labels that were genuinely computed.
   - **`interpretations`** - hedged, explicitly non-diagnostic readings of a real pattern in the observations (e.g. "a majority of messages carried negative sentiment, which *may suggest* ..."). Every interpretation carries the same standard caveat verbatim: *"This is a descriptive, pattern-based observation - not a psychological or medical assessment."* Interpretations are only ever generated when a real pattern crosses a defined threshold (e.g. at least 3 sentiment-analyzed messages with a clear majority) - a conversation with too little data produces an empty `interpretations` list, never a speculative one. Neither observations nor interpretations ever use psychological or medical diagnostic language (no "depression", "anxiety disorder", "PTSD", etc.) - this is enforced by what the generation logic is capable of producing in the first place (a fixed, reviewed set of templates), not by a runtime filter.

## What has been genuinely verified

Live, against a real running server: a fresh account's dashboard showed every section as a genuine empty state (zero counts, `null` latencies, empty label lists); after one real `/ask` call, the dashboard showed real numbers (1 conversation, real 8ms retrieval latency, a real "neutral" sentiment count, a real "unavailable" grounding-status count, a real registered model version with its genuine RAVDESS evaluation metrics attached) and the insights endpoint for that conversation showed a real timeline entry with real sentiment/intent and a correctly-empty `interpretations` list (too little data to cross any threshold). Automated tests (`tests/integration/test_analytics_dashboard.py`, `tests/integration/test_conversation_insights.py`) cover the empty-state, real-activity, per-user-isolation, and pipeline-health-reflects-real-config cases.
