"""NLP service boundary (sentiment/intent/NER/topics). Implemented in Phase 3."""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class Entity(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int


class NlpAnnotation(BaseModel):
    sentiment_label: str
    sentiment_score: float
    intent_label: str
    intent_confidence: float
    topics: list[str]
    entities: list[Entity]
    model_versions: dict[str, str]


class NlpAnalyzer(Protocol):
    async def analyze(self, text: str) -> NlpAnnotation: ...
