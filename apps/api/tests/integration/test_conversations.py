from __future__ import annotations

import pytest

REGISTER_PAYLOAD = {"email": "convo@voxmind.dev", "password": "correct-horse-battery-staple"}


async def _authed_headers(client) -> dict:
    await client.post("/api/v1/auth/register", json=REGISTER_PAYLOAD)
    login = await client.post("/api/v1/auth/login", json=REGISTER_PAYLOAD)
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.mark.asyncio
async def test_create_and_retrieve_conversation(client):
    headers = await _authed_headers(client)

    create_response = await client.post(
        "/api/v1/conversations", json={"title": "First session"}, headers=headers
    )
    assert create_response.status_code == 201
    conversation_id = create_response.json()["id"]
    assert create_response.json()["status"] == "active"

    get_response = await client.get(f"/api/v1/conversations/{conversation_id}", headers=headers)
    assert get_response.status_code == 200
    assert get_response.json()["title"] == "First session"


@pytest.mark.asyncio
async def test_list_conversations_returns_only_own_conversations(client):
    headers_a = await _authed_headers(client)
    await client.post("/api/v1/conversations", json={"title": "A's session"}, headers=headers_a)

    client.cookies.clear()
    other_credentials = {"email": "other@voxmind.dev", "password": "another-password-123"}
    await client.post("/api/v1/auth/register", json=other_credentials)
    login_b = await client.post("/api/v1/auth/login", json=other_credentials)
    headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}
    await client.post("/api/v1/conversations", json={"title": "B's session"}, headers=headers_b)

    list_a = await client.get("/api/v1/conversations", headers=headers_a)
    titles_a = [c["title"] for c in list_a.json()]
    assert titles_a == ["A's session"]

    list_b = await client.get("/api/v1/conversations", headers=headers_b)
    titles_b = [c["title"] for c in list_b.json()]
    assert titles_b == ["B's session"]


@pytest.mark.asyncio
async def test_cannot_access_another_users_conversation(client):
    headers_a = await _authed_headers(client)
    created = await client.post("/api/v1/conversations", json={"title": "private"}, headers=headers_a)
    conversation_id = created.json()["id"]

    client.cookies.clear()
    await client.post(
        "/api/v1/auth/register", json={"email": "intruder@voxmind.dev", "password": "another-password-123"}
    )
    login_b = await client.post(
        "/api/v1/auth/login", json={"email": "intruder@voxmind.dev", "password": "another-password-123"}
    )
    headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

    response = await client.get(f"/api/v1/conversations/{conversation_id}", headers=headers_b)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_message_persistence_and_ordering(client):
    headers = await _authed_headers(client)
    created = await client.post("/api/v1/conversations", json={"title": "chat"}, headers=headers)
    conversation_id = created.json()["id"]

    for content in ["hello", "how are you", "goodbye"]:
        response = await client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"role": "user", "content": content},
            headers=headers,
        )
        assert response.status_code == 201

    messages_response = await client.get(
        f"/api/v1/conversations/{conversation_id}/messages", headers=headers
    )
    assert messages_response.status_code == 200
    contents = [m["content"] for m in messages_response.json()]
    assert contents == ["hello", "how are you", "goodbye"]


@pytest.mark.asyncio
async def test_message_rejects_assistant_role_in_phase_1(client):
    """There is no real LLM/response-generation pipeline yet (Phase 4). The
    API must not accept a client-supplied 'assistant' message, since that
    would be indistinguishable from faking an AI response."""
    headers = await _authed_headers(client)
    created = await client.post("/api/v1/conversations", json={"title": "chat"}, headers=headers)
    conversation_id = created.json()["id"]

    response = await client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"role": "assistant", "content": "I am a real AI response"},
        headers=headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_deleting_conversation_cascades_to_messages(client):
    """Checks the actual row-level cascade behavior against the real
    database - not just that the API stops exposing the conversation. A
    fresh, short-lived session is opened only at the point of the check
    (rather than held open across the preceding HTTP calls), which avoids a
    known asyncpg/SQLAlchemy-async rough edge where an idle session shares a
    greenlet context with unrelated concurrent ASGI request handling.
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from voxmind.core.config import get_settings
    from voxmind.models.message import Message

    headers = await _authed_headers(client)
    created = await client.post("/api/v1/conversations", json={"title": "to delete"}, headers=headers)
    conversation_id = created.json()["id"]
    await client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"role": "user", "content": "will be cascade-deleted"},
        headers=headers,
    )

    delete_response = await client.delete(f"/api/v1/conversations/{conversation_id}", headers=headers)
    assert delete_response.status_code == 204

    check_engine = create_async_engine(get_settings().DATABASE_URL, poolclass=NullPool)
    session_factory = async_sessionmaker(bind=check_engine, expire_on_commit=False)
    async with session_factory() as session:
        remaining = (
            (await session.execute(select(Message).where(Message.session_id == conversation_id)))
            .scalars()
            .all()
        )
    await check_engine.dispose()
    assert remaining == []
