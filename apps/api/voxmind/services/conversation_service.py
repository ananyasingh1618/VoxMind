"""Conversation/message application service. Ownership checks live here so
every endpoint gets them for free instead of re-implementing the check.
"""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.exceptions import NotFoundError
from voxmind.models.conversation import Conversation
from voxmind.models.message import Message
from voxmind.repositories.conversation_repository import ConversationRepository
from voxmind.repositories.message_repository import MessageRepository


class ConversationService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._conversations = ConversationRepository(session)
        self._messages = MessageRepository(session)

    async def create(self, *, user_id: uuid.UUID, title: str | None) -> Conversation:
        conversation = await self._conversations.create(user_id=user_id, title=title)
        await self._session.commit()
        return conversation

    async def list_for_user(self, *, user_id: uuid.UUID) -> list[Conversation]:
        return await self._conversations.list_for_user(user_id)

    async def get_owned(self, *, conversation_id: uuid.UUID, user_id: uuid.UUID) -> Conversation:
        conversation = await self._conversations.get_by_id(conversation_id)
        # NotFoundError (not ForbiddenError) for both "doesn't exist" and "not
        # yours" - avoids leaking whether a given conversation ID exists.
        if conversation is None or conversation.user_id != user_id:
            raise NotFoundError("Conversation not found.")
        return conversation

    async def delete_owned(self, *, conversation_id: uuid.UUID, user_id: uuid.UUID) -> None:
        conversation = await self.get_owned(conversation_id=conversation_id, user_id=user_id)
        await self._conversations.delete(conversation)
        await self._session.commit()

    async def add_message(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, role: str, content: str
    ) -> Message:
        await self.get_owned(conversation_id=conversation_id, user_id=user_id)
        message = await self._messages.create(session_id=conversation_id, role=role, content=content)
        await self._session.commit()
        return message

    async def list_messages(self, *, conversation_id: uuid.UUID, user_id: uuid.UUID) -> list[Message]:
        await self.get_owned(conversation_id=conversation_id, user_id=user_id)
        return await self._messages.list_for_conversation(conversation_id)
