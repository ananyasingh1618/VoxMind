# ADR 0007: The reranker loads in float64, not float32

## The bug

`CrossEncoderReranker` (`services/retrieval/reranker.py`), which runs `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank retrieved chunks, was producing `NaN` for every score - genuine live/integration testing surfaced this as a Postgres error (`invalid input syntax for type json: Token "NaN" is invalid`) when `RetrievalService` tried to persist the retrieval trace, since `NaN` is a valid Python float but not valid JSON.

Investigation (not assumption) traced this to two distinct issues:

1. A real tokenization bug: the reranker's `_run()` originally called `tokenizer(pairs, ...)` with `pairs` as a list of `(query, passage)` tuples passed as the single positional argument. Hugging Face's fast tokenizers expect **two parallel lists** (`text`, `text_pair`) for sentence-pair tasks, not a list of tuples as the first argument - the call silently mis-tokenized instead of raising. Fixed by passing `tokenizer(queries, passages, ...)` as two parallel lists.
2. After fixing (1), scores were *still* `NaN`. Direct investigation (loading the model standalone, feeding it plain single-sentence input with no pairing at all, inspecting `output_hidden_states` layer by layer) showed the encoder's very first transformer block already produces `NaN` in `float32` on this machine's CPU backend (PyTorch 2.14.0, macOS arm64) - reproduced with correct, real, non-degenerate input and confirmed the model's own weights contain no `NaN`. Re-running the identical input and weights in `float64` produced clean, sensible logits (e.g. `-9.31` for an unrelated sentence, `+8.26` for a relevant one). This is a genuine float32 numerical-precision bug in this specific checkpoint on this environment's CPU backend, not a weights or tokenization problem.

## The fix

- `_run()` now tokenizes with `tokenizer(queries, passages, ...)` (two parallel lists) - the correct, documented API for sentence-pair models.
- The model is loaded with `dtype=torch.float64` instead of the default `float32`. The model is small (6 layers, hidden size 384, ~22M params), so the precision/cost tradeoff is negligible.
- As defense in depth (not a substitute for the real fix above), `RetrievalService` also filters any non-finite score out of `rerank_scores` before persisting the retrieval trace (`math.isfinite()` guard) - so a future, different numerical edge case degrades to "no rerank score for that chunk" rather than a hard failure writing to Postgres.

## Why this is the right level of fix

This is a real, reproducible bug in a Phase 4 component being implemented in this same pass - not a Phase 1-3 architecture change, and not a case of avoiding "fake" output by disabling reranking. The instruction to fix "a concrete bug" applies squarely here: reranking is explicitly required ("reranking where justified"), and silently leaving it broken (or worse, quietly disabling it to avoid the crash) would have meant claiming reranking works when it produced unusable output on every single call in this environment.

## Verification

After the fix: standalone reranker test (`CrossEncoderReranker.score()`) returns real, correctly-ordered scores for a relevant vs. irrelevant passage. The full `/ask` endpoint, exercised against a genuinely running server, now returns a non-null `rerank_score` in `retrieved_chunks` - see `docs/rag.md`'s live verification section.
