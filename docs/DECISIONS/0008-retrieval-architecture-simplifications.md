# ADR 0008: Two Phase 4 retrieval design simplifications

## 1. No separate `VectorStore` abstraction

`services/retrieval/interfaces.py` (written in Phase 0/1 as a forward-looking design note) sketched a `VectorStore` protocol (`add`/`search`/`delete`) as a layer distinct from `RetrievalProvider`, reasoning that the vector index might need to be swappable independently of retrieval logic.

Implementing it as a literal separate class would have meant wrapping `KnowledgeChunkRepository` in a second abstraction that adds no real behavior: a `KnowledgeChunk` row already stores its embedding at creation time (via `KnowledgeService`/`KnowledgeChunkRepository.create`), so there is no "add an embedding to an existing chunk later" operation to abstract over, and pgvector's index lives on the same table as the chunk's content, metadata, and lexical-search target - splitting them would require either duplicating chunk identity across two stores or reintroducing the same join `KnowledgeChunk`'s denormalization was designed to avoid.

Retrieval is instead implemented directly against `KnowledgeChunkRepository.search_vector()`/`search_lexical()` from `RetrievalService`. If a genuinely separate vector index (e.g. a dedicated vector database) is ever warranted, `KnowledgeChunkRepository`'s two search methods are the seam to swap - no call site outside `RetrievalService` depends on pgvector specifics.

## 2. Approximate token budgeting, not a real tokenizer

`services/llm/context_assembler.py` enforces `Settings.CONTEXT_TOKEN_BUDGET` using a fixed characters-per-token estimate (`Settings.CHARS_PER_TOKEN_ESTIMATE`, default 4), not a real tokenizer's exact count. Anthropic ships no offline tokenizer; `tiktoken` is OpenAI-specific and would give a wrong count for the Anthropic path. A single approximation shared by both providers is simpler and more honest than a per-provider exact count that's still wrong for a third provider added later.

This is documented as a budgeting heuristic in the module docstring and here - it is never used for billing, and the trimming logic (drop oldest recent turns → shorten retrieved excerpts → drop the summary → never trim the current question) is deliberately conservative so a slight over/under-estimate doesn't silently truncate the user's actual question.
