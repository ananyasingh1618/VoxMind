"""Real document ingestion against a real Postgres database: real chunking,
real sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2` via plain
transformers - no mocking), and a real pgvector column. Marked `real_model`
like Phase 2/3's Whisper/Wav2Vec2 tests - downloads and runs a genuine model,
slow, but needs no credentials.
"""
from __future__ import annotations

import pytest

CREDENTIALS = {"email": "knowledge-test@voxmind.dev", "password": "correct-horse-battery-staple"}

DOCUMENT_TEXT = (
    b"VoxMind is a multimodal conversational intelligence platform.\n\n"
    b"It combines speech transcription, emotion recognition, and retrieval-"
    b"augmented generation to build a deep understanding of a conversation.\n\n"
    b"The retrieval pipeline uses pgvector for vector search and PostgreSQL "
    b"full-text search for lexical search, fused with reciprocal rank fusion."
)


async def _authed_headers(client) -> dict:
    await client.post("/api/v1/auth/register", json=CREDENTIALS)
    login = await client.post("/api/v1/auth/login", json=CREDENTIALS)
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


async def _create_conversation(client, headers) -> str:
    response = await client.post("/api/v1/conversations", json={"title": "knowledge test"}, headers=headers)
    return response.json()["id"]


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_document_upload_and_real_ingestion_produces_searchable_chunks(client):
    headers = await _authed_headers(client)
    conversation_id = await _create_conversation(client, headers)

    upload = await client.post(
        f"/api/v1/conversations/{conversation_id}/documents",
        files={"file": ("voxmind.txt", DOCUMENT_TEXT, "text/plain")},
        headers=headers,
    )
    assert upload.status_code == 201
    document = upload.json()
    assert document["status"] == "pending"
    assert document["chunk_count"] == 0

    process = await client.post(
        f"/api/v1/conversations/{conversation_id}/documents/{document['id']}/process",
        headers=headers,
    )
    assert process.status_code == 201
    processed = process.json()
    assert processed["status"] == "completed"
    assert processed["chunk_count"] >= 1
    assert processed["error_message"] is None

    listing = await client.get(f"/api/v1/conversations/{conversation_id}/documents", headers=headers)
    assert listing.status_code == 200
    assert len(listing.json()) == 1


@pytest.mark.asyncio
async def test_unsupported_document_type_is_rejected(client):
    headers = await _authed_headers(client)
    conversation_id = await _create_conversation(client, headers)

    upload = await client.post(
        f"/api/v1/conversations/{conversation_id}/documents",
        files={"file": ("virus.exe", b"\x00\x01\x02", "application/x-msdownload")},
        headers=headers,
    )
    assert upload.status_code == 415


@pytest.mark.asyncio
async def test_document_too_large_is_rejected(client, monkeypatch):
    from voxmind.core.config import get_settings

    headers = await _authed_headers(client)
    conversation_id = await _create_conversation(client, headers)

    settings = get_settings()
    monkeypatch.setattr(settings, "MAX_DOCUMENT_UPLOAD_BYTES", 10)

    upload = await client.post(
        f"/api/v1/conversations/{conversation_id}/documents",
        files={"file": ("big.txt", b"x" * 1000, "text/plain")},
        headers=headers,
    )
    assert upload.status_code == 413
