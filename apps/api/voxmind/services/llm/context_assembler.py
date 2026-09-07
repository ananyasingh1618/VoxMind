"""Deterministic context assembly with an enforced token budget. This is the
only place a `ConversationContext` is constructed - the LLM never receives
raw conversation-database rows, only what this function selects and fits
within `Settings.CONTEXT_TOKEN_BUDGET`.

Token counting uses a documented, conservative characters-per-token
approximation (`Settings.CHARS_PER_TOKEN_ESTIMATE`) rather than a real
tokenizer, since not every configured provider ships one usable offline
(Anthropic has none; tiktoken is OpenAI-specific). This is called out
explicitly wherever it matters (docs/rag.md) - it is a budgeting heuristic,
never used for billing or exact truncation guarantees.

Trimming order when over budget (highest priority kept last):
  1. drop oldest recent-turn entries first
  2. shorten retrieved-chunk excerpts
  3. drop the conversation summary
The current user question and system instructions are never trimmed.
"""
from __future__ import annotations

from voxmind.core.config import Settings
from voxmind.services.llm.interfaces import ConversationContext
from voxmind.services.llm.schema import SYSTEM_INSTRUCTIONS


def _context_char_length(context: ConversationContext) -> int:
    import json

    return len(
        context.transcript
        + json.dumps(context.speaker_info, default=str)
        + json.dumps(context.emotion, default=str)
        + json.dumps(context.nlp, default=str)
        + json.dumps(context.incongruence, default=str)
        + json.dumps(context.memory, default=str)
        + json.dumps(context.retrieved_knowledge, default=str)
        + context.system_instructions
        + json.dumps(context.safety_context, default=str)
    )


def assemble_context(
    *,
    settings: Settings,
    transcript: str,
    speaker_info: list[dict],
    emotion: list[dict],
    nlp: dict | None,
    incongruence: list[dict],
    recent_turns: list[dict],
    conversation_summary: str | None,
    retrieved_knowledge: list[dict],
) -> tuple[ConversationContext, int]:
    """Returns (context, estimated_token_count). `recent_turns` must already
    be ordered oldest-first; the oldest entries are dropped first when
    trimming is needed."""
    budget_chars = settings.CONTEXT_TOKEN_BUDGET * settings.CHARS_PER_TOKEN_ESTIMATE

    turns = list(recent_turns)
    summary = conversation_summary
    knowledge = [dict(chunk) for chunk in retrieved_knowledge]

    def build() -> ConversationContext:
        return ConversationContext(
            transcript=transcript,
            speaker_info=speaker_info,
            emotion=emotion,
            nlp=nlp,
            incongruence=incongruence,
            memory={"recent_turns": turns, "summary": summary},
            retrieved_knowledge=knowledge,
            system_instructions=SYSTEM_INSTRUCTIONS,
            safety_context={"citations_must_reference_retrieved_chunk_ids": True},
        )

    context = build()

    # Step 1: drop oldest recent turns.
    while _context_char_length(context) > budget_chars and turns:
        turns.pop(0)
        context = build()

    # Step 2: shorten retrieved-chunk excerpts.
    excerpt_limit = 500
    while _context_char_length(context) > budget_chars and excerpt_limit > 100:
        excerpt_limit -= 100
        knowledge = [
            {**chunk, "text": str(chunk.get("text", ""))[:excerpt_limit]} for chunk in knowledge
        ]
        context = build()

    # Step 3: drop the summary as a last resort.
    if _context_char_length(context) > budget_chars and summary:
        summary = None
        context = build()

    final_char_length = _context_char_length(context)
    return context, max(1, final_char_length // settings.CHARS_PER_TOKEN_ESTIMATE)
