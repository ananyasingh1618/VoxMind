from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import get_current_user
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.conversation import ConversationCreate, ConversationOut
from voxmind.schemas.message import MessageCreate, MessageOut
from voxmind.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: ConversationCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ConversationOut:
    conversation = await ConversationService(session).create(user_id=current_user.id, title=body.title)
    return ConversationOut.model_validate(conversation)


@router.get("", response_model=list[ConversationOut])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> list[ConversationOut]:
    conversations = await ConversationService(session).list_for_user(user_id=current_user.id)
    return [ConversationOut.model_validate(c) for c in conversations]


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> ConversationOut:
    conversation = await ConversationService(session).get_owned(
        conversation_id=conversation_id, user_id=current_user.id
    )
    return ConversationOut.model_validate(conversation)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_conversation(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> None:
    await ConversationService(session).delete_owned(conversation_id=conversation_id, user_id=current_user.id)


@router.post(
    "/{conversation_id}/messages", response_model=MessageOut, status_code=status.HTTP_201_CREATED
)
async def create_message(
    conversation_id: uuid.UUID,
    body: MessageCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> MessageOut:
    message = await ConversationService(session).add_message(
        conversation_id=conversation_id, user_id=current_user.id, role=body.role, content=body.content
    )
    return MessageOut.model_validate(message)


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> list[MessageOut]:
    messages = await ConversationService(session).list_messages(
        conversation_id=conversation_id, user_id=current_user.id
    )
    return [MessageOut.model_validate(m) for m in messages]
