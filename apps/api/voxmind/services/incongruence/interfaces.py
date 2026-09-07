"""Semantic-vocal incongruence service boundary. Implemented in Phase 4.

Naming and framing are deliberate and approved: this is an analytical signal
describing divergence between spoken words (semantic signal) and vocal tone
(vocal signal) — e.g. flat/incongruent delivery of positive words can
indicate sarcasm, stress, or suppressed emotion. It is explicitly NOT a lie
detector, deception detector, or truthfulness predictor, and must never be
named, scored, or described as one anywhere in the codebase, API, or UI.
`signal_category` is a machine-readable guardrail against that framing
drifting back in under later implementation pressure.
"""
from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel


class IncongruenceSignal(BaseModel):
    incongruence_score: float
    semantic_signal: dict
    vocal_signal: dict
    confidence: float
    explanation: str
    signal_category: Literal["analytical_not_diagnostic"] = "analytical_not_diagnostic"


class IncongruenceAnalyzer(Protocol):
    async def analyze(self, semantic_signal: dict, vocal_signal: dict) -> IncongruenceSignal: ...
