"""Memory application service: short-term (recent turns, sent verbatim),
long-term (rolling `ConversationSummary`), and the trigger that decides when
enough un-summarized history has accrued to summarize it - `MemoryProvider`
from services/memory/interfaces.py, implemented for real here.

Summarization prefers a real configured LLM provider (produces a genuinely
better summary); if none is configured or the call fails, it falls back to
`summarize_extractive` (a real, deterministic algorithm - see
services/memory/extractive_summarizer.py), and `method` on the persisted row
honestly records which path ran. Never sends the *entire* conversation to
either path - only the un-summarized-and-not-recent slice, which is exactly
the point of summarizing.
"""
from __future__ import annotations

import uuid

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings
from voxmind.models.conversation_summary import ConversationSummary
from voxmind.models.message import Message
from voxmind.repositories.conversation_summary_repository import ConversationSummaryRepository
from voxmind.repositories.message_repository import MessageRepository
from voxmind.services.llm.interfaces import ConversationContext, LlmProvider
from voxmind.services.memory.extractive_summarizer import summarize_extractive

logger = structlog.get_logger(__name__)

SUMMARIZATION_SYSTEM_INSTRUCTIONS = (
    "You summarize conversation excerpts concisely and factually, preserving key facts, "
    "decisions, and open questions. Never add information that isn't present in the excerpt."
)


class MemoryService:
    def __init__(self, session: AsyncSession, *, settings: Settings, llm_provider: LlmProvider | None) -> None:
        self._session = session
        self._settings = settings
        self._llm_provider = llm_provider
        self._messages = MessageRepository(session)
        self._summaries = ConversationSummaryRepository(session)

    async def get_recent_turns(self, conversation_id: uuid.UUID) -> list[Message]:
        messages = await self._messages.list_for_conversation(conversation_id)
        return messages[-self._settings.MEMORY_RECENT_TURNS :]

    async def get_latest_summary(self, conversation_id: uuid.UUID) -> ConversationSummary | None:
        return await self._summaries.get_latest(conversation_id)

    async def summarize_if_needed(self, conversation_id: uuid.UUID) -> ConversationSummary | None:
        messages = await self._messages.list_for_conversation(conversation_id)
        latest_summary = await self._summaries.get_latest(conversation_id)
        covered_ids = set(latest_summary.covers_message_ids) if latest_summary else set()

        uncovered = [m for m in messages if str(m.id) not in covered_ids]
        recent_window = self._settings.MEMORY_RECENT_TURNS
        to_summarize = uncovered[:-recent_window] if len(uncovered) > recent_window else []

        if len(to_summarize) < self._settings.SUMMARY_TRIGGER_MESSAGE_COUNT:
            return None

        excerpt = "\n".join(f"{m.role}: {m.content}" for m in to_summarize)
        if latest_summary:
            excerpt = f"Previous summary: {latest_summary.summary_text}\n\nNew messages:\n{excerpt}"

        summary_text, method = await self._generate_summary(excerpt)
        row = await self._summaries.create(
            session_id=conversation_id,
            summary_text=summary_text,
            covers_message_ids=[m.id for m in to_summarize],
            method=method,
        )
        await self._session.commit()
        logger.info(
            "conversation_summarized",
            conversation_id=str(conversation_id),
            method=method,
            messages_covered=len(to_summarize),
        )
        return row

    async def _generate_summary(self, excerpt: str) -> tuple[str, str]:
        if self._llm_provider is not None:
            try:
                context = ConversationContext(
                    transcript=(
                        "Summarize the following conversation excerpt in 3-5 sentences:\n\n" + excerpt
                    ),
                    speaker_info=[],
                    emotion=[],
                    nlp=None,
                    incongruence=[],
                    memory=None,
                    retrieved_knowledge=[],
                    system_instructions=SUMMARIZATION_SYSTEM_INSTRUCTIONS,
                    safety_context={},
                )
                response = await self._llm_provider.generate(context)
                if response.answer.strip():
                    return response.answer.strip(), "llm_generated"
            except Exception as exc:  # noqa: BLE001 - any provider failure falls back, never fatal
                logger.warning("llm_summarization_failed_falling_back", error=str(exc))

        return summarize_extractive(excerpt, max_sentences=5), "extractive_fallback"
