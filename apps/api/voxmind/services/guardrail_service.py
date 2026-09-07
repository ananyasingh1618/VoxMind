"""Guardrail application service (Phase 6): the dedicated layer between the
LLM and everything downstream of it (a persisted assistant `Message`, TTS).

    LLM -> [structured/citation validation] -> [grounding, already computed
    by services/llm/grounding.py] -> [safety check] -> approved/modified/
    blocked -> (only then) a Message is created / TTS is ever called.

`RagService.generate_for_message()` is the only caller - both the typed
`/ask` endpoint and the Phase 5 voice loop go through it, so neither has a
path that reaches TTS or persistence with an unvalidated response. Every
call also filters retrieved chunks for prompt-injection patterns *before*
they're assembled into the LLM's context (`filter_retrieved_chunks()`),
which is the concrete implementation of "retrieval filtering" /
"prompt injection defenses for RAG".
"""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.repositories.guardrail_evaluation_repository import GuardrailEvaluationRepository
from voxmind.services.guardrails.interfaces import (
    GuardrailDecision,
    GuardrailResult,
    InjectionScanResult,
    ModerationProvider,
)
from voxmind.services.guardrails.output_validation import validate_structured_output
from voxmind.services.guardrails.prompt_injection import scan_for_injection
from voxmind.services.llm.interfaces import LlmResponse
from voxmind.services.retrieval.interfaces import RetrievedChunk

SAFE_REFUSAL_MESSAGE = (
    "I'm not able to provide a response to that request. If you have a different question, "
    "I'm happy to help."
)
NO_GROUNDED_ANSWER_MESSAGE = "I don't have enough grounded information to answer that."


class GuardrailService:
    def __init__(self, session: AsyncSession, *, moderation_provider: ModerationProvider) -> None:
        self._session = session
        self._moderation_provider = moderation_provider
        self._evaluations = GuardrailEvaluationRepository(session)

    def scan_user_input(self, text: str) -> InjectionScanResult:
        """Informational only - logged on the guardrail row, never used to
        block or alter the user's own message (see module docstring on
        services/guardrails/prompt_injection.py for why)."""
        return scan_for_injection(text)

    def filter_retrieved_chunks(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], list[str], list[dict]]:
        """Excludes any retrieved chunk whose text matches a prompt-injection
        pattern before it ever reaches context assembly. Returns (kept,
        filtered_chunk_ids, filtered_details)."""
        kept: list[RetrievedChunk] = []
        filtered_ids: list[str] = []
        filtered_details: list[dict] = []
        for chunk in chunks:
            scan = scan_for_injection(chunk.text)
            if scan.is_suspicious:
                filtered_ids.append(str(chunk.chunk_id))
                filtered_details.append(
                    {"chunk_id": str(chunk.chunk_id), "matched_patterns": scan.matched_patterns}
                )
            else:
                kept.append(chunk)
        return kept, filtered_ids, filtered_details

    async def evaluate_response(
        self,
        *,
        session_id: uuid.UUID,
        llm_generation_id: uuid.UUID | None,
        response: LlmResponse,
        grounding_status: str,
        filtered_chunk_ids: list[str],
        input_injection: InjectionScanResult,
    ) -> GuardrailResult:
        output_check = validate_structured_output(response, grounding_status=grounding_status)
        safety = await self._moderation_provider.moderate(response.answer or response.evidence_summary)

        reasons: list[str] = []
        decision = GuardrailDecision.APPROVED
        final_answer = response.answer

        if not output_check.is_valid:
            reasons.extend(output_check.issues)
            decision = GuardrailDecision.MODIFIED
            if not final_answer.strip():
                final_answer = NO_GROUNDED_ANSWER_MESSAGE

        if safety.is_unsafe:
            reasons.append(f"Safety check ({safety.method}) flagged: {', '.join(safety.categories)}")
            decision = GuardrailDecision.BLOCKED
            final_answer = SAFE_REFUSAL_MESSAGE

        await self._evaluations.create(
            session_id=session_id,
            llm_generation_id=llm_generation_id,
            decision=decision.value,
            reasons=reasons,
            input_injection_detected=input_injection.is_suspicious,
            input_injection_patterns=input_injection.matched_patterns,
            filtered_chunk_ids=filtered_chunk_ids,
            output_validation_issues=output_check.issues,
            safety_flagged=safety.is_unsafe,
            safety_categories=safety.categories,
            safety_method=safety.method,
            original_answer=response.answer,
            final_answer=final_answer,
        )
        await self._session.commit()

        return GuardrailResult(
            decision=decision,
            final_answer=final_answer,
            reasons=reasons,
            filtered_chunk_ids=filtered_chunk_ids,
            output_validation=output_check,
            safety=safety,
        )
