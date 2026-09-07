"""Real, deterministic prompt-injection pattern scanning - the same
regex-based-heuristic honesty standard as `services/nlp/intent_classifier.py`
(a genuine classifier, not a trained model, documented as such).

Applied to two different inputs for two different reasons:
- **Retrieved document chunks** (`services/rag_service.py`'s retrieval-
  filtering step): a chunk of an uploaded document has no legitimate reason
  to contain meta-instructions directed at the AI system, so any match here
  results in the chunk being excluded from context entirely before it ever
  reaches the LLM - see docs/guardrails.md's "retrieval filtering" section.
- **The user's own message** (informational only, never blocking): a real
  user question can innocently contain phrases like "how do I ignore
  exceptions in Python", so a match here is recorded on the
  `GuardrailEvaluation` row for transparency, not used to block or alter
  the user's own input.
"""
from __future__ import annotations

import re

from voxmind.services.guardrails.interfaces import InjectionScanResult

_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all)\b[^.\n]{0,20}\b"
        r"(instructions?|prompts?|rules?)\b",
        re.IGNORECASE,
    ),
    "role_override": re.compile(
        r"\byou are now\b|\bact as (an?|the)\b[^.\n]{0,30}\b(unrestricted|uncensored|jailbroken|dan)\b",
        re.IGNORECASE,
    ),
    "reveal_system_prompt": re.compile(
        r"\b(reveal|show|print|repeat|output)\b[^.\n]{0,20}\b(your |the )?(system prompt|instructions|"
        r"chain[- ]of[- ]thought|hidden prompt)\b",
        re.IGNORECASE,
    ),
    "fake_role_marker": re.compile(r"(^|\n)\s*(system|assistant)\s*:\s*", re.IGNORECASE),
    "jailbreak_keyword": re.compile(r"\bjailbreak\b|\bdo anything now\b", re.IGNORECASE),
    "new_instructions": re.compile(r"\bnew instructions?\s*:", re.IGNORECASE),
}


def scan_for_injection(text: str) -> InjectionScanResult:
    matched = [name for name, pattern in _PATTERNS.items() if pattern.search(text)]
    return InjectionScanResult(is_suspicious=bool(matched), matched_patterns=matched)
