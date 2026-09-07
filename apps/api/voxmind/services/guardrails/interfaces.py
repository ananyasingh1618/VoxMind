"""Guardrail service boundary (Phase 6): the layer every LLM answer passes
through before it becomes a persisted assistant `Message` or reaches TTS.

`GuardrailDecision` is the only outcome that leaves this layer -
`approved` (sent through unchanged), `modified` (something was substituted,
e.g. an empty/invalid answer replaced with an honest fallback sentence), or
`blocked` (replaced entirely with a safe refusal). There is no path from an
LLM's raw output to TTS or to a persisted `Message` that skips this decision.
"""
from __future__ import annotations

from enum import Enum
from typing import Protocol

from pydantic import BaseModel


class GuardrailDecision(str, Enum):
    APPROVED = "approved"
    MODIFIED = "modified"
    BLOCKED = "blocked"


class InjectionScanResult(BaseModel):
    is_suspicious: bool
    matched_patterns: list[str]


class SafetyScanResult(BaseModel):
    is_unsafe: bool
    categories: list[str]
    method: str  # "openai_moderation" | "keyword_fallback"


class OutputValidationResult(BaseModel):
    is_valid: bool
    issues: list[str]


class GuardrailResult(BaseModel):
    decision: GuardrailDecision
    final_answer: str
    reasons: list[str]
    filtered_chunk_ids: list[str]
    output_validation: OutputValidationResult
    safety: SafetyScanResult


class ModerationProvider(Protocol):
    async def moderate(self, text: str) -> SafetyScanResult: ...
