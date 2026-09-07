"""Deterministic, rule-based intent classification. This is a genuine
classifier (every input maps through the same explicit rules to the same
output every time) rather than a trained model - documented honestly as
such so `intent_confidence` is never mistaken for a calibrated ML
probability. Rules are checked in priority order; the first match wins.
"""
from __future__ import annotations

import re

_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b", re.IGNORECASE)
_GRATITUDE_RE = re.compile(r"\b(thanks|thank you|appreciate it)\b", re.IGNORECASE)
_COMMAND_RE = re.compile(
    r"^\s*(please\s+)?(show|list|find|search|summarize|explain|tell me|give me|create|delete|"
    r"generate|translate|calculate|compute)\b",
    re.IGNORECASE,
)
_WH_QUESTION_RE = re.compile(r"^\s*(who|what|when|where|why|how|which|whose)\b", re.IGNORECASE)
_YES_NO_QUESTION_RE = re.compile(
    r"^\s*(is|are|do|does|did|can|could|will|would|should|has|have)\b", re.IGNORECASE
)


def classify_intent(text: str) -> tuple[str, float]:
    """Returns (intent_label, confidence). Confidence is 1.0 for a clean
    rule match and 0.5 for the "statement" fallback (no rule matched), a
    deliberately simple two-tier scheme rather than an invented decimal."""
    stripped = text.strip()
    has_question_mark = stripped.endswith("?")

    if _GREETING_RE.match(stripped):
        return "greeting", 1.0
    if _GRATITUDE_RE.search(stripped):
        return "gratitude", 1.0
    if _COMMAND_RE.match(stripped):
        return "command", 1.0
    if has_question_mark or _WH_QUESTION_RE.match(stripped) or _YES_NO_QUESTION_RE.match(stripped):
        return "question", 1.0
    return "statement", 0.5
