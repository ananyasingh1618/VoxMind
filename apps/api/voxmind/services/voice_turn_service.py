"""Phase 5 real-time voice loop orchestrator: audio capture -> STT ->
speaker/emotion/NLP analysis -> memory/RAG -> LLM -> TTS -> voice response.
This is the only place that whole loop is wired together; every stage it
calls is a real, already-implemented Phase 2-4 service - nothing here is
reimplemented, only sequenced and instrumented.

`run()` is an async generator that yields one real progress event per stage
as that stage genuinely completes (never simulated/timed fake progress) -
`api/v1/endpoints/voice.py` turns these into Server-Sent Events. Stage names
map directly to the UI states the frontend renders: `processing` (STT),
`thinking` (emotion/NLP/incongruence analysis), `retrieving` (RAG
retrieval), `generating` (LLM), `speaking` (TTS). `listening` is a
frontend-only state (before any audio is sent) and is never emitted here.

Real interruption - found and fixed for real, not assumed: an earlier
version of this method polled `await request.is_disconnected()` between
stages, on the assumption that a client disconnecting mid-stream would
otherwise go unnoticed. Live testing (killing a client connection mid-
request) proved that assumption wrong in an interesting way: Starlette's
`StreamingResponse` already detects a client disconnect on its own and
cancels the serving task directly - the manual polling checks never even
got a chance to run, because a real `asyncio.CancelledError` was delivered
straight into whatever `await` the generator happened to be suspended at.
The correct fix is the opposite of more polling: catch that cancellation
once, at the top level, and use it to do cleanup - not to detect the
disconnect (the runtime already did that). See docs/voice.md and
docs/DECISIONS/0009 for the full story, including why the cleanup itself
must run inside `anyio.CancelScope(shield=True)` (a `VoiceTurn` update
issued from inside a cancelled scope would itself be immediately cancelled
before it could reach the database, without shielding).

Latency instrumentation: `stage_latencies_ms` on the persisted `VoiceTurn`
records genuinely measured (`time.monotonic()`) durations for exactly the
buckets requested - `stt_ms`, `analysis_ms`, `retrieval_ms`, `llm_ms`,
`tts_ms`, `total_ms` - never estimated.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import anyio
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request

from voxmind.core.config import Settings
from voxmind.core.exceptions import NotFoundError
from voxmind.models.message import Message
from voxmind.models.voice_turn import VoiceTurn
from voxmind.repositories.emotion_prediction_repository import EmotionPredictionRepository
from voxmind.repositories.message_repository import MessageRepository
from voxmind.repositories.transcript_repository import TranscriptRepository
from voxmind.repositories.voice_turn_repository import VoiceTurnRepository
from voxmind.services.audio_service import AudioService
from voxmind.services.conversation_service import ConversationService
from voxmind.services.emotion_service import EmotionService
from voxmind.services.incongruence_service import IncongruenceService
from voxmind.services.nlp_service import NlpService
from voxmind.services.rag_service import RagService
from voxmind.services.storage.interfaces import StorageBackend
from voxmind.services.tts.interfaces import TextToSpeechProvider

logger = structlog.get_logger(__name__)


class VoiceTurnService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        settings: Settings,
        storage: StorageBackend,
        audio_service: AudioService,
        emotion_service: EmotionService,
        nlp_service: NlpService,
        incongruence_service: IncongruenceService,
        rag_service: RagService,
        tts_provider: TextToSpeechProvider | None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._storage = storage
        self._audio_service = audio_service
        self._emotion_service = emotion_service
        self._nlp_service = nlp_service
        self._incongruence_service = incongruence_service
        self._rag_service = rag_service
        self._tts_provider = tts_provider
        self._conversations = ConversationService(session)
        self._messages = MessageRepository(session)
        self._transcripts = TranscriptRepository(session)
        self._emotion_predictions = EmotionPredictionRepository(session)
        self._voice_turns = VoiceTurnRepository(session)

    async def run(
        self,
        *,
        request: Request,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        filename: str | None,
        raw_bytes: bytes,
    ) -> AsyncIterator[dict[str, Any]]:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        turn = await self._voice_turns.create(session_id=conversation_id)
        await self._session.commit()
        log = logger.bind(conversation_id=str(conversation_id), voice_turn_id=str(turn.id))

        latencies: dict[str, int] = {}
        total_started = time.monotonic()

        try:
            async for event in self._run_pipeline(
                request=request,
                conversation_id=conversation_id,
                user_id=user_id,
                filename=filename,
                raw_bytes=raw_bytes,
                turn=turn,
                latencies=latencies,
                total_started=total_started,
                log=log,
            ):
                yield event
        except asyncio.CancelledError:
            # The client disconnected mid-pipeline and the ASGI server has
            # already cancelled this task - not something we detect
            # ourselves. Record it while we still can: a plain `await` here
            # would immediately raise the same CancelledError again (the
            # enclosing scope is already cancelled), so the cleanup write is
            # shielded from it.
            with anyio.CancelScope(shield=True):
                await self._voice_turns.mark_interrupted(
                    turn, error_message="Client disconnected before the voice turn finished."
                )
                await self._session.commit()
                log.info("voice_turn_interrupted", stage_latencies_ms=latencies)
            raise

    async def _run_pipeline(
        self,
        *,
        request: Request,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        filename: str | None,
        raw_bytes: bytes,
        turn: VoiceTurn,
        latencies: dict[str, int],
        total_started: float,
        log: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"stage": "processing", "status": "start", "voice_turn_id": str(turn.id)}

        # --- STT ------------------------------------------------------
        stt_started = time.monotonic()
        try:
            asset = await self._audio_service.upload_audio(
                conversation_id=conversation_id, user_id=user_id, filename=filename, raw_bytes=raw_bytes
            )
            job = await self._audio_service.process_audio(
                conversation_id=conversation_id, user_id=user_id, audio_asset_id=asset.id
            )
        except Exception as exc:  # noqa: BLE001 - a real, reportable STT failure, never fabricated
            latencies["stt_ms"] = round((time.monotonic() - stt_started) * 1000)
            await self._voice_turns.mark_failed(turn, error_code="stt_failed", error_message=str(exc))
            await self._session.commit()
            log.warning("voice_turn_stt_failed", error=str(exc))
            yield {"stage": "error", "status": "failed", "error_code": "stt_failed", "error_message": str(exc)}
            return
        latencies["stt_ms"] = round((time.monotonic() - stt_started) * 1000)

        if job.status != "completed" or job.message_id is None:
            await self._voice_turns.mark_failed(
                turn, error_code="stt_failed", error_message=job.error_message or "Transcription failed."
            )
            await self._session.commit()
            yield {
                "stage": "error",
                "status": "failed",
                "error_code": "stt_failed",
                "error_message": job.error_message or "Transcription failed.",
            }
            return

        user_message = await self._messages.get_by_id(job.message_id)
        assert user_message is not None  # just created above by AudioService in this same transaction

        yield {
            "stage": "processing",
            "status": "completed",
            "transcript": user_message.content,
            "latency_ms": latencies["stt_ms"],
        }

        # --- Analysis: emotion + NLP + incongruence (best-effort) -----
        yield {"stage": "thinking", "status": "start"}
        analysis_started = time.monotonic()
        emotion_context, incongruence_context = await self._run_analysis(
            conversation_id=conversation_id, user_id=user_id, message=user_message, log=log
        )
        latencies["analysis_ms"] = round((time.monotonic() - analysis_started) * 1000)
        yield {"stage": "thinking", "status": "completed", "latency_ms": latencies["analysis_ms"]}

        # --- Memory / RAG / LLM (shared with the text ask() flow) -----
        yield {"stage": "retrieving", "status": "start"}
        rag_answer = await self._rag_service.generate_for_message(
            conversation_id=conversation_id,
            user_id=user_id,
            user_message=user_message,
            emotion=emotion_context,
            incongruence=incongruence_context,
        )
        latencies["retrieval_ms"] = rag_answer.retrieval_ms
        latencies["llm_ms"] = rag_answer.llm_ms
        yield {
            "stage": "retrieving",
            "status": "completed",
            "retrieved_chunk_count": len(rag_answer.retrieved_chunks),
            "latency_ms": latencies["retrieval_ms"],
        }
        yield {
            "stage": "generating",
            "status": "completed" if rag_answer.assistant_message else "unavailable",
            "answer": rag_answer.generation.answer,
            "grounding_status": rag_answer.generation.grounding_status,
            "error_message": rag_answer.generation.error_message,
            "latency_ms": latencies["llm_ms"],
        }
        if rag_answer.guardrail_result is not None:
            guardrail = rag_answer.guardrail_result
            yield {
                "stage": "guardrail",
                "status": guardrail.decision.value,
                "reasons": guardrail.reasons,
                "final_answer": guardrail.final_answer,
            }

        # --- TTS --------------------------------------------------------
        audio_storage_key: str | None = None
        audio_content_type: str | None = None
        audio_provider: str | None = None
        audio_duration_ms: int | None = None
        tts_error: str | None = None
        latencies["tts_ms"] = 0

        if rag_answer.assistant_message is None:
            yield {"stage": "speaking", "status": "unavailable", "reason": "No answer was generated."}
        elif self._tts_provider is None:
            yield {"stage": "speaking", "status": "unavailable", "reason": "No TTS provider is configured."}
        else:
            yield {"stage": "speaking", "status": "start"}
            tts_started = time.monotonic()
            speech_text = rag_answer.assistant_message.content[: self._settings.VOICE_TURN_MAX_ANSWER_CHARS]
            try:
                tts_result = await self._tts_provider.synthesize(speech_text)
                audio_content_type = tts_result.content_type
                audio_provider = tts_result.provider
                audio_duration_ms = tts_result.duration_ms
                extension = "wav" if "wav" in tts_result.content_type else "mp3"
                audio_storage_key = f"conversations/{conversation_id}/voice-turns/{turn.id}/response.{extension}"
                await self._storage.upload(
                    audio_storage_key, tts_result.audio_bytes, content_type=tts_result.content_type
                )
            except Exception as exc:  # noqa: BLE001 - TTS failure is reported, never a fake clip
                tts_error = str(exc)
                log.warning("voice_turn_tts_failed", error=tts_error)
            latencies["tts_ms"] = round((time.monotonic() - tts_started) * 1000)
            if audio_storage_key:
                yield {
                    "stage": "speaking",
                    "status": "completed",
                    "audio_duration_ms": audio_duration_ms,
                    "latency_ms": latencies["tts_ms"],
                }
            else:
                yield {"stage": "speaking", "status": "failed", "error_message": tts_error}

        latencies["total_ms"] = round((time.monotonic() - total_started) * 1000)

        if rag_answer.assistant_message is None or audio_storage_key is None:
            status = "partial"
        else:
            status = "completed"

        await self._voice_turns.mark_finished(
            turn,
            status=status,
            user_message_id=user_message.id,
            assistant_message_id=rag_answer.assistant_message.id if rag_answer.assistant_message else None,
            llm_generation_id=rag_answer.generation.id,
            audio_storage_key=audio_storage_key,
            audio_content_type=audio_content_type,
            audio_provider=audio_provider,
            audio_duration_ms=audio_duration_ms,
            stage_latencies_ms=latencies,
            error_message=tts_error,
        )
        await self._session.commit()
        log.info("voice_turn_completed", status=status, stage_latencies_ms=latencies)

        yield {
            "stage": "done",
            "status": status,
            "voice_turn_id": str(turn.id),
            "user_message_id": str(user_message.id),
            "assistant_message_id": (
                str(rag_answer.assistant_message.id) if rag_answer.assistant_message else None
            ),
            "answer": rag_answer.generation.answer,
            "citations": rag_answer.generation.citations,
            "grounding_status": rag_answer.generation.grounding_status,
            "audio_url": (
                f"/api/v1/conversations/{conversation_id}/voice-turns/{turn.id}/audio"
                if audio_storage_key
                else None
            ),
            "stage_latencies_ms": latencies,
        }

    async def _run_analysis(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, message: Message, log: Any
    ) -> tuple[list[dict], list[dict]]:
        """Emotion + NLP + incongruence are best-effort: a real per-turn
        emotion model may not be registered, and any single failure here
        must never block the LLM from answering. Returns (emotion_context,
        incongruence_context) - both real, both possibly empty."""
        try:
            await self._emotion_service.process_message(
                conversation_id=conversation_id, user_id=user_id, message_id=message.id
            )
        except Exception as exc:  # noqa: BLE001
            log.info("voice_turn_emotion_unavailable", error=str(exc))

        try:
            await self._nlp_service.process_message(
                conversation_id=conversation_id, user_id=user_id, message_id=message.id
            )
        except Exception as exc:  # noqa: BLE001
            log.info("voice_turn_nlp_failed", error=str(exc))

        emotion_context: list[dict] = []
        try:
            turns = await self._transcripts.get_aligned_turns(message.id)
            for turn in turns:
                predictions = await self._emotion_predictions.list_for_turn(turn.id)
                if predictions:
                    emotion_context.append(
                        {
                            "speaker_label": turn.speaker_label,
                            "predicted_label": predictions[0].predicted_label,
                            "confidence": predictions[0].confidence,
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            log.info("voice_turn_emotion_context_unavailable", error=str(exc))

        incongruence_context: list[dict] = []
        try:
            signals = await self._incongruence_service.analyze_message(
                conversation_id=conversation_id, user_id=user_id, message_id=message.id
            )
            incongruence_context = [
                {
                    "incongruence_score": s.incongruence_score,
                    "confidence": s.confidence,
                    "explanation": s.explanation,
                }
                for s in signals
            ]
        except Exception as exc:  # noqa: BLE001
            log.info("voice_turn_incongruence_unavailable", error=str(exc))

        return emotion_context, incongruence_context

    async def get_turn(self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, turn_id: uuid.UUID):
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        turn = await self._voice_turns.get_by_id(turn_id)
        if turn is None or turn.session_id != conversation_id:
            raise NotFoundError("Voice turn not found.")
        return turn

    async def get_audio(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, turn_id: uuid.UUID
    ) -> tuple[bytes, str]:
        turn = await self.get_turn(conversation_id=conversation_id, user_id=user_id, turn_id=turn_id)
        if not turn.audio_storage_key:
            raise NotFoundError("This voice turn has no synthesized audio.")
        audio_bytes = await self._storage.download(turn.audio_storage_key)
        return audio_bytes, turn.audio_content_type or "application/octet-stream"
