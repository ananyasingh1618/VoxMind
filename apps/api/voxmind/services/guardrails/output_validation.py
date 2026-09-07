"""Real structural/invariant checks on an LLM's structured response - beyond
what Pydantic's field types already enforce (`services/llm/schema.py`'s
forced tool-call shape). These catch a response that is *well-formed JSON*
but still internally inconsistent or unsafe to relay as-is, e.g. a
"grounded" answer with no actual text, or a leaked chain-of-thought marker
(this project's Phase 0 constraint against ever exposing one - see
docs/DECISIONS and services/llm/interfaces.py's `evidence_summary` field).
"""
from __future__ import annotations

import re

from voxmind.services.guardrails.interfaces import OutputValidationResult
from voxmind.services.llm.interfaces import LlmResponse

_REASONING_LEAK_PATTERNS = [
    re.compile(r"<\s*thinking\s*>", re.IGNORECASE),
    re.compile(r"\bchain[- ]of[- ]thought\b", re.IGNORECASE),
    re.compile(r"\blet me think (step by step|through this)\b", re.IGNORECASE),
]


def validate_structured_output(response: LlmResponse, *, grounding_status: str) -> OutputValidationResult:
    issues: list[str] = []

    if not (0.0 <= response.confidence <= 1.0):
        issues.append(f"confidence {response.confidence} is outside the valid [0.0, 1.0] range")

    if grounding_status == "grounded" and not response.answer.strip():
        issues.append("grounding_status is 'grounded' but the answer text is empty")

    if not response.answer.strip() and not response.evidence_summary.strip():
        issues.append("both answer and evidence_summary are empty")

    for pattern in _REASONING_LEAK_PATTERNS:
        if pattern.search(response.answer) or pattern.search(response.evidence_summary):
            issues.append("response appears to contain an internal-reasoning/chain-of-thought marker")
            break

    return OutputValidationResult(is_valid=not issues, issues=issues)
