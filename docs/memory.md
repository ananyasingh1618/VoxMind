# Conversation memory (Phase 4)

VoxMind never sends an unrestricted dump of a conversation's message history to the LLM. `MemoryService` (`services/memory_service.py`) maintains two distinct, explicitly separated representations, both consumed only through `services/llm/context_assembler.py` (see `docs/rag.md`):

## Short-term memory: recent turns

The last `Settings.MEMORY_RECENT_TURNS` (default 8) messages, sent to the context assembler verbatim (role + content). This is genuinely bounded - a 500-message conversation still only contributes its 8 most recent turns here, never the whole history.

## Long-term memory: rolling summaries

`ConversationSummary` rows are immutable once created - a new summarization pass never overwrites an old one, it inserts a new row covering the newly-accrued messages (plus, in its *input* text, the previous summary's text) so the full history of what the assistant "knew" at each point stays auditable.

**Explicit criterion for what triggers a summary**: once the number of messages that are (a) not already covered by the latest summary and (b) not part of the current short-term recent-turns window reaches `Settings.SUMMARY_TRIGGER_MESSAGE_COUNT` (default 12), `MemoryService.summarize_if_needed()` summarizes exactly that slice - never the entire conversation, and never messages already covered or still in the short-term window (they don't need summarizing yet). This is checked automatically after every `RagService.ask()` call, and can be triggered manually via `POST /conversations/{id}/memory/summarize` for inspection.

**Two summarization methods, honestly labeled** (`ConversationSummary.method`):

- `llm_generated` - if a real LLM provider is configured (`services/llm/factory.py`), the same provider used for answering is asked to produce a concise, factual summary of the un-summarized excerpt.
- `extractive_fallback` - if no provider is configured, or the LLM call fails, `services/memory/extractive_summarizer.py::summarize_extractive()` runs instead: a real, deterministic frequency-weighted sentence-extraction algorithm (score each sentence by the sum of its non-stopword words' document frequency, normalized by sentence length; keep the top-N highest-scoring sentences in their original order - the well-established "Luhn-style" extractive summarization technique). This is never presented as equivalent in quality to an LLM-generated summary; the `method` field records exactly which path ran, on every row.

Both paths only ever see the specific un-summarized slice being summarized (plus the prior summary's text as context) - never the full conversation database.

## Retrieval

`GET /conversations/{id}/memory` returns the current recent-turns window and latest summary (if any) for inspection. `MemoryContext`/`ConversationSummary` (`services/memory/interfaces.py`) define the structured shape; nothing here ever exposes a model's private reasoning - only the summary text itself, which is factual conversation content, not chain-of-thought.

## What has been genuinely verified

`tests/integration/test_memory_service.py` exercises, against a real Postgres database: the recent-turns window boundary, the no-summary-before-threshold case, a real extractive-fallback summarization run (asserting the real `Luhn`-style algorithm produces non-empty, threshold-respecting output), and an LLM-backed summarization path using a stub provider (proving `method="llm_generated"` is correctly recorded when a provider is available - the stub exists only to avoid a real network call in a unit-scoped test, not to fabricate what a real provider would return).
