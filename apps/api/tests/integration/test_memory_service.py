"""Memory retrieval and summarization-trigger tests against a real Postgres
database. No LLM provider is configured in these tests, so summarization
exercises the real, deterministic extractive fallback - never a fake or
LLM-shaped summary text.
"""
from __future__ import annotations

import pytest

from voxmind.core.config import get_settings
from voxmind.models.conversation import Conversation
from voxmind.repositories.conversation_repository import ConversationRepository
from voxmind.repositories.message_repository import MessageRepository
from voxmind.repositories.user_repository import UserRepository
from voxmind.services.memory_service import MemoryService


async def _make_conversation(db_session) -> Conversation:
    users = UserRepository(db_session)
    user = await users.create(email="memory-test@voxmind.dev", hashed_password="not-a-real-hash")
    conversations = ConversationRepository(db_session)
    conversation = await conversations.create(user_id=user.id, title="memory test")
    await db_session.commit()
    return conversation


@pytest.mark.asyncio
async def test_recent_turns_window_returns_only_the_configured_count(db_session):
    conversation = await _make_conversation(db_session)
    messages = MessageRepository(db_session)
    for i in range(20):
        await messages.create(session_id=conversation.id, role="user", content=f"message {i}")
    await db_session.commit()

    settings = get_settings()
    service = MemoryService(db_session, settings=settings, llm_provider=None)
    recent = await service.get_recent_turns(conversation.id)

    assert len(recent) == settings.MEMORY_RECENT_TURNS
    assert recent[-1].content == "message 19"  # most recent kept, chronological order
    await db_session.commit()


@pytest.mark.asyncio
async def test_no_summary_before_threshold_is_reached(db_session):
    conversation = await _make_conversation(db_session)
    messages = MessageRepository(db_session)
    for i in range(3):
        await messages.create(session_id=conversation.id, role="user", content=f"short message {i}")
    await db_session.commit()

    settings = get_settings()
    service = MemoryService(db_session, settings=settings, llm_provider=None)
    summary = await service.summarize_if_needed(conversation.id)
    assert summary is None
    await db_session.commit()


@pytest.mark.asyncio
async def test_summarization_triggers_once_threshold_is_crossed_using_extractive_fallback(db_session):
    conversation = await _make_conversation(db_session)
    messages = MessageRepository(db_session)
    settings = get_settings()
    total_needed = settings.SUMMARY_TRIGGER_MESSAGE_COUNT + settings.MEMORY_RECENT_TURNS + 2
    for i in range(total_needed):
        await messages.create(
            session_id=conversation.id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"This is a real sentence about topic number {i}. It has real words in it.",
        )
    await db_session.commit()

    service = MemoryService(db_session, settings=settings, llm_provider=None)
    summary = await service.summarize_if_needed(conversation.id)

    assert summary is not None
    assert summary.method == "extractive_fallback"
    assert len(summary.summary_text) > 0
    assert len(summary.covers_message_ids) >= settings.SUMMARY_TRIGGER_MESSAGE_COUNT

    latest = await service.get_latest_summary(conversation.id)
    assert latest.id == summary.id
    await db_session.commit()


@pytest.mark.asyncio
async def test_llm_summarizer_is_used_when_a_provider_is_configured(db_session):
    from voxmind.services.llm.interfaces import ConversationContext, LlmResponse

    class _StubLlmProvider:
        provider_name = "stub"
        model_name = "stub-v1"

        async def generate(self, context: ConversationContext) -> LlmResponse:
            return LlmResponse(
                answer="Stub summary of the conversation.",
                citations=[],
                confidence=0.9,
                evidence_summary="stub",
            )

    conversation = await _make_conversation(db_session)
    messages = MessageRepository(db_session)
    settings = get_settings()
    total_needed = settings.SUMMARY_TRIGGER_MESSAGE_COUNT + settings.MEMORY_RECENT_TURNS + 2
    for i in range(total_needed):
        await messages.create(session_id=conversation.id, role="user", content=f"content {i}")
    await db_session.commit()

    service = MemoryService(db_session, settings=settings, llm_provider=_StubLlmProvider())
    summary = await service.summarize_if_needed(conversation.id)

    assert summary.method == "llm_generated"
    assert summary.summary_text == "Stub summary of the conversation."
    await db_session.commit()
