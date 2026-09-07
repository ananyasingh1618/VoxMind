"""RAG application service: the end-to-end flow behind `POST
/conversations/{id}/ask` **and** the Phase 5 real-time voice loop
(`VoiceTurnService`) - memory -> retrieval -> context assembly -> real LLM
generation (or honest "unavailable") -> citation validation -> grounding ->
**the Phase 6 guardrail layer** -> persistence. Every sub-step is a real
implementation from nlp_service/memory_service/retrieval_service/
services.llm/guardrail_service.

`generate_for_message()` is the shared core: `ask()` (text entry point)
creates a `Message` from a typed question and calls it; `VoiceTurnService`
(audio entry point) already has a real transcribed `Message` (from
`AudioService`) plus real emotion/incongruence signals and calls it
directly, so a spoken question and a typed question produce an answer
through the exact same grounding/citation/guardrail/persistence logic - and,
as of Phase 6, so neither entry point has a path to a persisted assistant
`Message` (and therefore to TTS, which only ever reads that message's
content) that skips the guardrail layer.

If no LLM provider is configured (`services.llm.factory.build_llm_provider`
returned `None`), this never fabricates an answer: no assistant `Message` is
created, and the persisted `LlmGeneration` row honestly records
`grounding_status="unavailable"` with an explanatory `error_message` - the
same pattern Phase 2/3 established for diarization/emotion.
"""
from __future__ import annotations

import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.core.config import Settings
from voxmind.models.llm_generation import LlmGeneration
from voxmind.models.message import Message
from voxmind.repositories.llm_generation_repository import LlmGenerationRepository
from voxmind.repositories.nlp_annotation_repository import NlpAnnotationRepository
from voxmind.services.conversation_service import ConversationService
from voxmind.services.guardrail_service import GuardrailService
from voxmind.services.guardrails.interfaces import GuardrailResult
from voxmind.services.llm.context_assembler import assemble_context
from voxmind.services.llm.grounding import validate_citations
from voxmind.services.llm.interfaces import LlmProvider
from voxmind.services.memory_service import MemoryService
from voxmind.services.nlp_service import NlpService
from voxmind.services.retrieval.interfaces import RetrievedChunk
from voxmind.services.retrieval_service import RetrievalService


class RagAnswer:
    """Plain result object returned to the API layer - not persisted
    verbatim, just a convenient bundle of everything the caller needs to
    shape its response. `retrieval_ms`/`llm_ms` are genuinely measured
    per-stage durations (0 when that stage didn't run), consumed by
    `VoiceTurnService` for its `stage_latencies_ms` instrumentation."""

    def __init__(
        self,
        *,
        user_message: Message,
        assistant_message: Message | None,
        retrieved_chunks: list[RetrievedChunk],
        retrieval_result_id: uuid.UUID,
        generation: LlmGeneration,
        retrieval_ms: int,
        llm_ms: int,
        guardrail_result: GuardrailResult | None = None,
    ) -> None:
        self.user_message = user_message
        self.assistant_message = assistant_message
        self.retrieved_chunks = retrieved_chunks
        self.retrieval_result_id = retrieval_result_id
        self.generation = generation
        self.retrieval_ms = retrieval_ms
        self.llm_ms = llm_ms
        self.guardrail_result = guardrail_result


class RagService:
    def __init__(
        self,
        session: AsyncSession,
        *,
        settings: Settings,
        llm_provider: LlmProvider | None,
        nlp_service: NlpService,
        memory_service: MemoryService,
        retrieval_service: RetrievalService,
        guardrail_service: GuardrailService,
    ) -> None:
        self._session = session
        self._settings = settings
        self._llm_provider = llm_provider
        self._nlp_service = nlp_service
        self._memory_service = memory_service
        self._retrieval_service = retrieval_service
        self._guardrail_service = guardrail_service
        self._conversations = ConversationService(session)
        self._generations = LlmGenerationRepository(session)
        self._nlp_annotations = NlpAnnotationRepository(session)

    async def ask(self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, question: str) -> RagAnswer:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)
        user_message = await self._conversations.add_message(
            conversation_id=conversation_id, user_id=user_id, role="user", content=question
        )
        return await self.generate_for_message(
            conversation_id=conversation_id, user_id=user_id, user_message=user_message
        )

    async def generate_for_message(
        self,
        *,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID,
        user_message: Message,
        emotion: list[dict] | None = None,
        incongruence: list[dict] | None = None,
    ) -> RagAnswer:
        """Runs memory -> retrieval -> context assembly -> LLM -> grounding
        for an ALREADY-EXISTING user message (typed or transcribed from real
        audio) and persists the result. `emotion`/`incongruence` let a
        caller that already computed real per-turn signals (the voice loop)
        fold them into the assembled context; `ask()`'s typed-text path has
        neither, so both default to empty."""
        nlp_annotation = await self._resolve_nlp_annotation(
            conversation_id=conversation_id, user_id=user_id, user_message=user_message
        )

        recent_turns = await self._memory_service.get_recent_turns(conversation_id)
        latest_summary = await self._memory_service.get_latest_summary(conversation_id)
        recent_turn_dicts = [
            {"role": m.role, "content": m.content} for m in recent_turns if m.id != user_message.id
        ]

        retrieval_started = time.monotonic()
        raw_retrieved_chunks, retrieval_result_id = await self._retrieval_service.retrieve(
            conversation_id=conversation_id, query_message_id=user_message.id, query=user_message.content
        )
        retrieval_ms = round((time.monotonic() - retrieval_started) * 1000)

        # Retrieval filtering (Phase 6): a retrieved document chunk has no
        # legitimate reason to contain meta-instructions directed at the AI
        # system - any that do are excluded here, before the LLM ever sees
        # them, and the exclusion is logged on the guardrail evaluation.
        retrieved_chunks, filtered_chunk_ids, _filtered_details = self._guardrail_service.filter_retrieved_chunks(
            raw_retrieved_chunks
        )
        input_injection = self._guardrail_service.scan_user_input(user_message.content)

        retrieved_knowledge = [
            {"chunk_id": str(c.chunk_id), "document_title": c.citation, "text": c.text}
            for c in retrieved_chunks
        ]

        context, token_estimate = assemble_context(
            settings=self._settings,
            transcript=user_message.content,
            speaker_info=[],
            emotion=emotion or [],
            nlp=nlp_annotation,
            incongruence=incongruence or [],
            recent_turns=recent_turn_dicts,
            conversation_summary=latest_summary.summary_text if latest_summary else None,
            retrieved_knowledge=retrieved_knowledge,
        )
        retrieved_chunk_ids = {str(c.chunk_id) for c in retrieved_chunks}

        if self._llm_provider is None:
            generation = await self._generations.create(
                session_id=conversation_id,
                query_message_id=user_message.id,
                answer_message_id=None,
                retrieval_result_id=retrieval_result_id,
                provider=self._settings.LLM_PROVIDER,
                model_name=None,
                context_token_estimate=token_estimate,
                answer="",
                citations=[],
                confidence=0.0,
                evidence_summary="",
                grounding_status="unavailable",
                grounding_details={},
                error_message=(
                    "No LLM provider is configured (missing API key, or LLM_PROVIDER=local_dev). "
                    "See docs/rag.md for required environment variables."
                ),
                latency_ms=0,
            )
            await self._session.commit()
            await self._memory_service.summarize_if_needed(conversation_id)
            return RagAnswer(
                user_message=user_message,
                assistant_message=None,
                retrieved_chunks=retrieved_chunks,
                retrieval_result_id=retrieval_result_id,
                generation=generation,
                retrieval_ms=retrieval_ms,
                llm_ms=0,
            )

        llm_started = time.monotonic()
        try:
            response = await self._llm_provider.generate(context)
            error_message = None
        except Exception as exc:  # noqa: BLE001 - a provider failure is reported, never fabricated
            llm_ms = round((time.monotonic() - llm_started) * 1000)
            generation = await self._generations.create(
                session_id=conversation_id,
                query_message_id=user_message.id,
                answer_message_id=None,
                retrieval_result_id=retrieval_result_id,
                provider=getattr(self._llm_provider, "provider_name", self._settings.LLM_PROVIDER),
                model_name=getattr(self._llm_provider, "model_name", None),
                context_token_estimate=token_estimate,
                answer="",
                citations=[],
                confidence=0.0,
                evidence_summary="",
                grounding_status="unavailable",
                grounding_details={},
                error_message=f"LLM provider call failed: {exc}",
                latency_ms=llm_ms,
            )
            await self._session.commit()
            return RagAnswer(
                user_message=user_message,
                assistant_message=None,
                retrieved_chunks=retrieved_chunks,
                retrieval_result_id=retrieval_result_id,
                generation=generation,
                retrieval_ms=retrieval_ms,
                llm_ms=llm_ms,
            )
        llm_ms = round((time.monotonic() - llm_started) * 1000)

        valid_citations, grounding_status, grounding_details = validate_citations(
            response.citations, retrieved_chunk_ids
        )

        # Persisted first (answer_message_id filled in below) so the
        # guardrail evaluation - which references it by id - always has a
        # real LlmGeneration row to point at, capturing the model's raw
        # output before any guardrail modification.
        generation = await self._generations.create(
            session_id=conversation_id,
            query_message_id=user_message.id,
            answer_message_id=None,
            retrieval_result_id=retrieval_result_id,
            provider=getattr(self._llm_provider, "provider_name", self._settings.LLM_PROVIDER),
            model_name=getattr(self._llm_provider, "model_name", None),
            context_token_estimate=token_estimate,
            answer=response.answer,
            raw_model_answer=response.answer,
            citations=[c.model_dump() for c in valid_citations],
            confidence=response.confidence,
            evidence_summary=response.evidence_summary,
            grounding_status=grounding_status,
            grounding_details=grounding_details,
            error_message=error_message,
            latency_ms=llm_ms,
        )
        await self._session.commit()

        # Guardrail layer (Phase 6): the ONLY path from here to a persisted
        # assistant Message (and therefore to TTS, which only reads that
        # message's content) - never the raw model response directly.
        guardrail_result = await self._guardrail_service.evaluate_response(
            session_id=conversation_id,
            llm_generation_id=generation.id,
            response=response,
            grounding_status=grounding_status,
            filtered_chunk_ids=filtered_chunk_ids,
            input_injection=input_injection,
        )

        assistant_message = await self._conversations.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=guardrail_result.final_answer,
        )
        # `answer` is overwritten with the guardrail-approved text - this is
        # the only "answer" any API response or SSE event ever reads.
        # `raw_model_answer` (set above, untouched from here) keeps the
        # original for audit only.
        generation.answer = guardrail_result.final_answer
        generation.answer_message_id = assistant_message.id
        await self._session.commit()

        await self._memory_service.summarize_if_needed(conversation_id)

        return RagAnswer(
            user_message=user_message,
            assistant_message=assistant_message,
            retrieved_chunks=retrieved_chunks,
            retrieval_result_id=retrieval_result_id,
            generation=generation,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            guardrail_result=guardrail_result,
        )

    async def _resolve_nlp_annotation(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID, user_message: Message
    ) -> dict | None:
        """Reuses an already-computed `NlpAnnotation` if one exists for this
        message (the voice loop runs NLP as its own "analysis" stage before
        calling `generate_for_message`) - otherwise computes one inline
        (the typed-text `ask()` path, where NLP hasn't run yet). Never runs
        it twice for the same message."""
        existing = await self._nlp_annotations.get_for_message(user_message.id)
        if existing is not None:
            return {
                "sentiment_label": existing.sentiment_label,
                "sentiment_score": existing.sentiment_score,
                "intent_label": existing.intent_label,
                "intent_confidence": existing.intent_confidence,
                "topics": existing.topics,
            }
        try:
            nlp_row = await self._nlp_service.process_message(
                conversation_id=conversation_id, user_id=user_id, message_id=user_message.id
            )
        except Exception:  # noqa: BLE001 - NLP enrichment is best-effort, never blocks an answer
            return None
        return {
            "sentiment_label": nlp_row.sentiment_label,
            "sentiment_score": nlp_row.sentiment_score,
            "intent_label": nlp_row.intent_label,
            "intent_confidence": nlp_row.intent_confidence,
            "topics": nlp_row.topics,
        }
