"""The Phase 5 real-time voice loop entry point: `POST
/conversations/{id}/voice-turns` accepts a recorded audio blob and streams
real progress as Server-Sent Events - each event is emitted the moment a
real backend stage genuinely completes (see `VoiceTurnService.run()`), never
simulated/timed fake progress. `GET .../voice-turns/{id}/audio` serves the
synthesized speech back for playback.
"""
from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.api.deps import get_current_user, get_settings, get_storage_backend, get_tts_provider
from voxmind.api.v1.endpoints.audio import get_audio_service
from voxmind.api.v1.endpoints.emotion import get_emotion_service
from voxmind.api.v1.endpoints.incongruence import get_incongruence_service
from voxmind.api.v1.endpoints.nlp import get_nlp_service
from voxmind.api.v1.endpoints.rag import get_rag_service
from voxmind.core.config import Settings
from voxmind.core.exceptions import AudioTooLargeError
from voxmind.core.rate_limit import rate_limit_by_user
from voxmind.db.session import get_db
from voxmind.models.user import User
from voxmind.schemas.voice import VoiceTurnOut
from voxmind.services.audio_service import AudioService
from voxmind.services.emotion_service import EmotionService
from voxmind.services.incongruence_service import IncongruenceService
from voxmind.services.nlp_service import NlpService
from voxmind.services.rag_service import RagService
from voxmind.services.storage.interfaces import StorageBackend
from voxmind.services.tts.interfaces import TextToSpeechProvider
from voxmind.services.voice_turn_service import VoiceTurnService

router = APIRouter(prefix="/conversations/{conversation_id}", tags=["voice"])


def get_voice_turn_service(
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(get_storage_backend),
    audio_service: AudioService = Depends(get_audio_service),
    emotion_service: EmotionService = Depends(get_emotion_service),
    nlp_service: NlpService = Depends(get_nlp_service),
    incongruence_service: IncongruenceService = Depends(get_incongruence_service),
    rag_service: RagService = Depends(get_rag_service),
    tts_provider: TextToSpeechProvider | None = Depends(get_tts_provider),
) -> VoiceTurnService:
    return VoiceTurnService(
        session,
        settings=settings,
        storage=storage,
        audio_service=audio_service,
        emotion_service=emotion_service,
        nlp_service=nlp_service,
        incongruence_service=incongruence_service,
        rag_service=rag_service,
        tts_provider=tts_provider,
    )


def _sse_event(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, default=str)}\n\n".encode()


@router.post(
    "/voice-turns",
    dependencies=[Depends(rate_limit_by_user("expensive", "RATE_LIMIT_EXPENSIVE_PER_WINDOW"))],
)
async def create_voice_turn(
    request: Request,
    conversation_id: uuid.UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    service: VoiceTurnService = Depends(get_voice_turn_service),
) -> StreamingResponse:
    """Streams the real target loop (audio capture -> STT -> analysis ->
    memory/RAG -> LLM -> TTS) as Server-Sent Events. See docs/voice.md for
    the exact event shapes and the UI states they drive.
    """
    raw_bytes = await file.read()
    if len(raw_bytes) > settings.MAX_AUDIO_UPLOAD_BYTES:
        raise AudioTooLargeError(f"Audio upload exceeds the {settings.MAX_AUDIO_UPLOAD_BYTES} byte limit.")
    filename = file.filename

    async def event_stream():
        try:
            async for event in service.run(
                request=request,
                conversation_id=conversation_id,
                user_id=current_user.id,
                filename=filename,
                raw_bytes=raw_bytes,
            ):
                yield _sse_event(event)
        except asyncio.CancelledError:
            # The client disconnected - VoiceTurnService.run() already
            # recorded status="interrupted" (see its CancelledError handler).
            # There is no connection left to send an "error" event to, and
            # cancellation must propagate, never be swallowed.
            raise
        except Exception as exc:  # noqa: BLE001 - never let a raw traceback leak into the SSE stream
            yield _sse_event({"stage": "error", "status": "failed", "error_message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/voice-turns/{turn_id}", response_model=VoiceTurnOut)
async def get_voice_turn(
    conversation_id: uuid.UUID,
    turn_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: VoiceTurnService = Depends(get_voice_turn_service),
) -> VoiceTurnOut:
    turn = await service.get_turn(conversation_id=conversation_id, user_id=current_user.id, turn_id=turn_id)
    return VoiceTurnOut.model_validate(turn)


@router.get("/voice-turns/{turn_id}/audio")
async def get_voice_turn_audio(
    conversation_id: uuid.UUID,
    turn_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: VoiceTurnService = Depends(get_voice_turn_service),
) -> Response:
    audio_bytes, content_type = await service.get_audio(
        conversation_id=conversation_id, user_id=current_user.id, turn_id=turn_id
    )
    return Response(content=audio_bytes, media_type=content_type)
