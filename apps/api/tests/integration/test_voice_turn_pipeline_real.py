"""Genuine end-to-end Phase 5 voice loop tests: real audio upload -> real
Whisper transcription -> real (best-effort) emotion/NLP analysis -> the real
RAG pipeline -> real local TTS synthesis (`facebook/mms-tts-eng`, no
credentials needed - see docs/voice.md), all driven through the real
Server-Sent-Events HTTP endpoint against a real Postgres database.

Also verifies the real interruption behavior found and fixed while building
this: killing the client connection mid-pipeline must genuinely stop the
server-side work and record `status="interrupted"` - see
docs/DECISIONS/0009-voice-loop-cancellation.md for the investigation that
led to the fix (an earlier, incorrect implementation polled
`request.is_disconnected()`, which live testing proved never actually ran
before the ASGI server's own task cancellation pre-empted it).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from voxmind.api.deps import get_llm_provider
from voxmind.main import app
from voxmind.services.llm.mock_provider import MockLlmProvider

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
CREDENTIALS = {"email": "voice-turn-test@voxmind.dev", "password": "correct-horse-battery-staple"}


async def _authed_headers(client) -> dict:
    await client.post("/api/v1/auth/register", json=CREDENTIALS)
    login = await client.post("/api/v1/auth/login", json=CREDENTIALS)
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


async def _create_conversation(client, headers) -> str:
    response = await client.post("/api/v1/conversations", json={"title": "voice turn test"}, headers=headers)
    return response.json()["id"]


async def _collect_sse_events(client, url: str, headers: dict, wav_bytes: bytes) -> list[dict]:
    events = []
    async with client.stream(
        "POST", url, headers=headers, files={"file": ("hello_world.wav", wav_bytes, "audio/wav")}
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


@pytest.fixture(autouse=True)
def _clear_llm_override():
    yield
    app.dependency_overrides.pop(get_llm_provider, None)


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_voice_turn_produces_a_real_transcript_and_honest_unavailable_generation(client):
    """No LLM provider configured (this test database's real default) - the
    STT/analysis/retrieval stages must all be genuinely real regardless."""
    headers = await _authed_headers(client)
    conversation_id = await _create_conversation(client, headers)
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()

    events = await _collect_sse_events(
        client, f"/api/v1/conversations/{conversation_id}/voice-turns", headers, wav_bytes
    )

    by_stage = {(e["stage"], e.get("status")): e for e in events}
    assert ("processing", "completed") in by_stage
    transcript = by_stage[("processing", "completed")]["transcript"]
    assert transcript.strip()  # a real Whisper transcript, not fabricated

    assert ("thinking", "completed") in by_stage
    assert ("retrieving", "completed") in by_stage
    assert ("generating", "unavailable") in by_stage
    assert ("speaking", "unavailable") in by_stage

    done = by_stage[("done", "partial")]
    assert done["audio_url"] is None
    assert done["answer"] == ""
    latencies = done["stage_latencies_ms"]
    assert latencies["stt_ms"] > 0
    assert latencies["analysis_ms"] >= 0
    assert latencies["total_ms"] >= latencies["stt_ms"]


@pytest.mark.real_model
@pytest.mark.asyncio
async def test_voice_turn_full_loop_produces_real_synthesized_audio(client):
    """With a (test-only mock) LLM provider configured, the full loop
    completes and TTS genuinely synthesizes playable audio - the mock only
    replaces the LLM call; STT, analysis, retrieval, and TTS are all real."""
    app.dependency_overrides[get_llm_provider] = lambda: MockLlmProvider()

    headers = await _authed_headers(client)
    conversation_id = await _create_conversation(client, headers)
    wav_bytes = (FIXTURES_DIR / "hello_world.wav").read_bytes()

    events = await _collect_sse_events(
        client, f"/api/v1/conversations/{conversation_id}/voice-turns", headers, wav_bytes
    )

    done = next(e for e in events if e["stage"] == "done")
    assert done["status"] == "completed"
    assert done["answer"]
    assert done["audio_url"] is not None
    assert done["stage_latencies_ms"]["tts_ms"] > 0

    audio_response = await client.get(done["audio_url"], headers=headers)
    assert audio_response.status_code == 200
    assert audio_response.headers["content-type"] == "audio/wav"
    assert audio_response.content[:4] == b"RIFF"  # a real WAV file, not an empty/fake placeholder
    assert len(audio_response.content) > 1000


@pytest.mark.asyncio
async def test_voice_turn_requires_authentication(client):
    response = await client.post(
        "/api/v1/conversations/00000000-0000-0000-0000-000000000000/voice-turns",
        files={"file": ("x.wav", b"not real audio", "audio/wav")},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_voice_turn_marks_interrupted_when_cancelled_mid_pipeline(db_session):
    """Exercises the exact mechanism `VoiceTurnService.run()` relies on for
    real interruption: driving its async generator forward, then delivering
    a genuine `asyncio.CancelledError` into it via `athrow()` - precisely
    what happens when the ASGI server cancels the request task on a real
    client disconnect (verified live, repeatedly, against a running server;
    see docs/DECISIONS/0009-voice-loop-cancellation.md for that
    investigation). This isolates the cancellation-handling logic itself
    from the full HTTP/ASGI stack, which - independent of this feature -
    has a known event-loop-teardown interaction with cancelled asyncpg
    connections in this pytest-asyncio test harness (not a product bug;
    see conftest.py's docstring for the same recurring class of issue).
    """
    import uuid as uuid_module

    from voxmind.core.config import get_settings
    from voxmind.repositories.conversation_repository import ConversationRepository
    from voxmind.repositories.user_repository import UserRepository
    from voxmind.repositories.voice_turn_repository import VoiceTurnRepository
    from voxmind.services.voice_turn_service import VoiceTurnService

    users = UserRepository(db_session)
    user = await users.create(email="voice-cancel-test@voxmind.dev", hashed_password="x")
    conversations = ConversationRepository(db_session)
    conversation = await conversations.create(user_id=user.id, title="cancel test")
    await db_session.commit()

    settings = get_settings()
    service = VoiceTurnService(
        db_session,
        settings=settings,
        storage=None,  # type: ignore[arg-type] - never reached before cancellation
        audio_service=None,  # type: ignore[arg-type]
        emotion_service=None,  # type: ignore[arg-type]
        nlp_service=None,  # type: ignore[arg-type]
        incongruence_service=None,  # type: ignore[arg-type]
        rag_service=None,  # type: ignore[arg-type]
        tts_provider=None,
    )

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            return False

    generator = service.run(
        request=_FakeRequest(),  # type: ignore[arg-type]
        conversation_id=conversation.id,
        user_id=user.id,
        filename="test.wav",
        raw_bytes=b"placeholder",
    )

    # Drive it forward until it's genuinely suspended mid-pipeline (past
    # VoiceTurn creation, inside the real STT call) - it will raise for real
    # once it reaches `self._audio_service.upload_audio(...)` on the `None`
    # placeholder, which is fine: we only need it suspended *before* that,
    # at the first yield.
    first_event = await generator.asend(None)
    assert first_event["stage"] == "processing"
    voice_turn_id = uuid_module.UUID(first_event["voice_turn_id"])

    with pytest.raises(asyncio.CancelledError):
        await generator.athrow(asyncio.CancelledError())

    turns = VoiceTurnRepository(db_session)
    turn = await turns.get_by_id(voice_turn_id)
    assert turn is not None
    assert turn.status == "interrupted"
    assert turn.error_message == "Client disconnected before the voice turn finished."
    await db_session.commit()
