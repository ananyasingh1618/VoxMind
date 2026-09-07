"""`NlpAnalyzer` implementation: combines real HF sentiment/NER models with
deterministic topic/intent extraction into one `NlpAnnotation`. See
docs/nlp.md for exactly which parts are neural models versus deterministic
heuristics - never blurred together as if all four were equally "ML".
"""
from __future__ import annotations

from voxmind.services.nlp.interfaces import NlpAnnotation
from voxmind.services.nlp.intent_classifier import classify_intent
from voxmind.services.nlp.ner_provider import HuggingFaceNerProvider
from voxmind.services.nlp.sentiment_provider import HuggingFaceSentimentProvider
from voxmind.services.nlp.topic_extractor import extract_topics


class RealNlpAnalyzer:
    def __init__(
        self, sentiment_provider: HuggingFaceSentimentProvider, ner_provider: HuggingFaceNerProvider
    ) -> None:
        self._sentiment_provider = sentiment_provider
        self._ner_provider = ner_provider

    async def analyze(self, text: str) -> NlpAnnotation:
        sentiment = await self._sentiment_provider.analyze(text)
        entities = await self._ner_provider.extract(text)
        intent_label, intent_confidence = classify_intent(text)
        topics = extract_topics(text)

        return NlpAnnotation(
            sentiment_label=sentiment.label,
            sentiment_score=sentiment.score,
            intent_label=intent_label,
            intent_confidence=intent_confidence,
            topics=topics,
            entities=entities,
            model_versions={
                "sentiment": self._sentiment_provider.model_id,
                "ner": self._ner_provider.model_id,
                "intent": "rule-based-v1",
                "topics": "keyword-frequency-v1",
            },
        )
