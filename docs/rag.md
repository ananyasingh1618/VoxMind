# Retrieval-augmented generation (Phase 4)

Real document ingestion, real hybrid retrieval, and a real (credential-gated) LLM provider abstraction with application-side citation validation and grounding - no fabricated retrieval, no hardcoded answers, no invented citations.

## Knowledge ingestion

Documents are uploaded and processed per-conversation (same ownership model as audio assets):

1. `POST /conversations/{id}/documents` (multipart) - real magic-byte-independent format check (`.txt`, `.md`, `.pdf`; `Content-Type` sniffing falls back to the file extension since browsers are inconsistent about `.md`'s declared type), size-limited (`MAX_DOCUMENT_UPLOAD_BYTES`, default 10 MB). Stores the raw file via `StorageBackend` and creates a `KnowledgeDocument` row (`status="pending"`).
2. `POST /conversations/{id}/documents/{document_id}/process` - dispatches the real pipeline through the approved, unmodified `TaskRunner` contract:
   - `DocumentIngestionStage`: downloads the raw file, parses it (`.txt`/`.md` decoded directly; `.pdf` via `pypdf`'s real per-page text extraction - a scanned/image-only PDF genuinely yields little or no text, surfaced honestly as a short/empty document rather than OCR'd or faked), and chunks it (`services/knowledge/chunking.py` - deterministic, paragraph-aware packing up to `CHUNK_SIZE_CHARS` with `CHUNK_OVERLAP_CHARS` overlap for any paragraph that has to be hard-split).
   - `ChunkEmbeddingStage`, dispatched once per chunk (mirroring Phase 3's per-turn embedding dispatch): a real sentence embedding via `sentence-transformers/all-MiniLM-L6-v2`, loaded with plain Hugging Face Transformers (`AutoModel`/`AutoTokenizer`, attention-mask-aware mean pooling + L2 normalization) - the `sentence-transformers` package itself is never a dependency, the same "load the base encoder directly" pattern as Phase 3's `HuggingFaceWav2Vec2Provider`.
   - Each embedding is stored on its `KnowledgeChunk` row as a real `pgvector` `vector(384)` column.
3. `GET /conversations/{id}/documents` / `GET /conversations/{id}/documents/{document_id}` - list/inspect ingestion status (`pending`/`processing`/`completed`/`failed`, with `chunk_count` and, on failure, a real `error_message` - never a fabricated success).

## Retrieval: real hybrid search

`RetrievalService.retrieve()` (`services/retrieval_service.py`), given a query message and its text:

1. **Vector search** - `KnowledgeChunkRepository.search_vector()`: real pgvector cosine-distance nearest-neighbor search (`embedding <=> :query_embedding`, via `pgvector.sqlalchemy`'s `cosine_distance()`), backed by an `ivfflat` index (`ix_knowledge_chunks_embedding_cosine`, `lists=10` - sized for demo/portfolio data volumes, not production scale).
2. **Lexical search** - `KnowledgeChunkRepository.search_lexical()`: genuine PostgreSQL full-text search - `to_tsvector('english', content) @@ plainto_tsquery('english', :query)`, ranked with `ts_rank_cd` (cover-density ranking), backed by a functional GIN index (`ix_knowledge_chunks_content_tsv`). This is real PostgreSQL FTS terminology and mechanism, not a "BM25-like" approximation.
3. **Hybrid fusion** - `services/retrieval/hybrid.py`: deterministic Reciprocal Rank Fusion (`score = Σ 1/(k + rank)` over every ranked list a candidate appears in, `k = Settings.HYBRID_RRF_K`, default 60 - the standard RRF constant), not a hand-tuned weighted average.
4. **Reranking** (optional, `Settings.RERANK_ENABLED`, default on) - a real cross-encoder, `cross-encoder/ms-marco-MiniLM-L-6-v2`, scores each (query, candidate) pair jointly (unlike the independent-scoring bi-encoder embeddings used in step 1). See [DECISIONS/0007](DECISIONS/0007-reranker-float64-precision-bug.md) for a real numerical-precision bug found and fixed in this component (the model must be loaded in `float64`, not the default `float32`, on this environment's CPU backend - verified via direct investigation, not assumed). If reranking is disabled or the model fails to load, chunks pass through with `rerank_score: null` - retrieval still works on hybrid fusion alone, never a fabricated score.
5. The final `RETRIEVAL_TOP_K` (default 5) chunks are returned as `RetrievedChunk`s (`chunk_id`, `document_id`, `text`, `citation`, and every component score), and the **entire fusion trace** - every candidate considered, not just the selected top-k, with `vector_rank`/`lexical_rank`/`hybrid_score`/`rerank_score`/`selected` - is persisted as one `retrieval_results` row, so a retrieval run is inspectable and reproducible after the fact.

If no chunks exist yet for the conversation, retrieval honestly returns an empty result (a `retrieval_results` row with `items: []`) rather than skipping persistence or fabricating a match.

## LLM provider abstraction

`services/llm/interfaces.py` (`LlmProvider.generate(ConversationContext) -> LlmResponse`) is implemented by two real providers, chosen via `LLM_PROVIDER`:

- `AnthropicLlmProvider` (`services/llm/anthropic_provider.py`) - the real Anthropic Messages API, structured output forced via a single tool definition + `tool_choice` (the model can only reply by calling `emit_structured_response`, never free prose) - parsing is a JSON read of the tool call's `input`, never regex/string-scraping.
- `OpenAiLlmProvider` (`services/llm/openai_provider.py`) - the real OpenAI Chat Completions API, the same structured-output-via-forced-function-call approach and the same schema, so switching providers never changes the shape of the response the rest of the app receives.

`services/llm/factory.py::build_llm_provider()` returns **`None`** - not a fake provider - when no provider is configured (`LLM_PROVIDER=local_dev`, the default) or the configured provider's API key is missing. This mirrors the "unavailable, never fabricated" pattern Phase 2/3 established for diarization/emotion: `RagService.ask()` checks for `None` and persists an honest `grounding_status="unavailable"` `LlmGeneration` with no assistant message created, rather than calling a provider that would fail or synthesizing an answer.

**Required environment variables** for a real provider: `ANTHROPIC_API_KEY` (Anthropic) or `OPENAI_API_KEY` (OpenAI) - both are developer-console API keys with their own billing; a Claude Pro/Max or ChatGPT Plus chat subscription does not grant API access (see `docs/architecture.md`'s LLM strategy section). `LLM_MODEL`/`OPENAI_MODEL` select the model; neither is required to have a real key configured to run every credential-independent part of Phase 4 (ingestion, retrieval, ranking, grounding logic) - only the final generation step needs one.

A test-only `MockLlmProvider` (`services/llm/mock_provider.py`) exists solely so the RAG pipeline's context-assembly/citation-validation/grounding logic can be exercised deterministically in tests without a real key - it is gated by `Settings.MOCK_LLM=true`, which the existing Phase 0 validator (`_validate_invariants`) already refuses outside `ENV=test`. It genuinely echoes back only real `chunk_id`s it was actually given, never an invented one.

## Context assembly and memory budgeting

`services/llm/context_assembler.py::assemble_context()` is the **only** place a `ConversationContext` is built - the LLM never receives a raw dump of the conversation database. It assembles: the current question, short-term memory (recent turns, verbatim), long-term memory (the latest `ConversationSummary`, see `docs/memory.md`), any NLP/emotion/incongruence signals for the current turn, and the retrieved knowledge chunks - then enforces `Settings.CONTEXT_TOKEN_BUDGET` (approximate, see [DECISIONS/0008](DECISIONS/0008-retrieval-architecture-simplifications.md)) by trimming in a fixed, conservative order: drop the oldest recent turns first, then shorten retrieved-chunk excerpts, then drop the summary as a last resort - the current question and system instructions are never trimmed.

## Citations and grounding

The model is instructed (`services/llm/schema.py::SYSTEM_INSTRUCTIONS`) to cite only `chunk_id`s it was actually given and never invent one - but it is never *trusted* to comply. `services/llm/grounding.py::validate_citations()` checks every returned citation's `chunk_id` against the set of `chunk_id`s that were genuinely retrieved for that query; any citation referencing an id that wasn't retrieved is **dropped before it is ever persisted or returned to the client** - a fabricated citation cannot reach the API surface even if a provider ignored its instructions. `grounding_status` is computed from this check, never self-reported by the model:

| Status | Meaning |
|---|---|
| `grounded` | every citation returned was valid, and at least one was returned |
| `partially_grounded` | some citations were valid, some were not (the invalid ones were dropped) |
| `ungrounded` | no citations were returned, all returned citations were invalid, or nothing was retrieved at all (there is no basis to call anything "grounded" in a document that was never retrieved) |
| `unavailable` | no LLM provider is configured, or the provider call failed |

`LlmGeneration.grounding_details` persists the raw check (`citations_returned`, `citations_valid`, `invalid_citation_ids`, `retrieved_chunk_count`) alongside the status.

## The `/ask` endpoint

`POST /conversations/{id}/ask` (`{question}`) is the single entry point tying every piece above together: creates the user `Message` → runs NLP on it (best-effort; a NLP failure doesn't block the answer) → loads memory → retrieves knowledge → assembles context → calls the configured LLM provider (or reports unavailable) → validates citations → computes grounding → creates the assistant `Message` (only on a successful, non-`unavailable` generation) → persists the `LlmGeneration` → triggers a summarization check. Returns `AskResponse`: the user/assistant messages, the retrieval trace, and the full `Generation` record - everything the UI needs to render "question → retrieved sources → ranked evidence → answer → citations" (`apps/web/src/components/conversation/KnowledgePanel.tsx`).

## What has been genuinely verified in this environment

- **Real, end-to-end, via a live running server** (not just the test client): document upload → real parsing/chunking → real MiniLM embeddings → real pgvector cosine search + real PostgreSQL full-text search + real RRF fusion → real cross-encoder reranking (`rerank_score` genuinely computed, see ADR 0007) → context assembly → `/ask` correctly reporting `grounding_status="unavailable"` with no LLM key configured, and `grounding_status="grounded"` with valid citations when the test-only mock provider is enabled (`MOCK_LLM=true`).
- **Not verified**: a real Anthropic/OpenAI API call. This environment has no `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` configured, and none was fabricated to force a "successful" demo - `AnthropicLlmProvider`/`OpenAiLlmProvider` are implemented and unit-tested (schema parsing, tool-choice construction) but their real network call has not been exercised end-to-end here. Configuring either key makes the real path immediately reachable with no code change.
