"""Conversation insights application service (Phase 6): builds a real,
per-message intelligence timeline plus a small set of explainable summary
insights - every one traceable back to a real persisted signal, never
invented. See schemas/insights.py for the observation/interpretation
distinction this service is required to keep - `_build_interpretations()`
is the only place interpretive (hedged, non-diagnostic) text is generated,
and only when a real pattern in the observations actually supports it.
"""
from __future__ import annotations

import statistics
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from voxmind.repositories.emotion_prediction_repository import EmotionPredictionRepository
from voxmind.repositories.incongruence_signal_repository import IncongruenceSignalRepository
from voxmind.repositories.message_repository import MessageRepository
from voxmind.repositories.nlp_annotation_repository import NlpAnnotationRepository
from voxmind.repositories.transcript_repository import TranscriptRepository
from voxmind.schemas.insights import (
    ConversationInsights,
    Interpretation,
    Observation,
    TimelineEmotionEntry,
    TimelineEntry,
    TimelineIncongruenceEntry,
)
from voxmind.services.conversation_service import ConversationService

HIGH_MISMATCH_THRESHOLD = 0.6


class InsightsService:
    def __init__(self, session: AsyncSession) -> None:
        self._conversations = ConversationService(session)
        self._messages = MessageRepository(session)
        self._nlp_annotations = NlpAnnotationRepository(session)
        self._transcripts = TranscriptRepository(session)
        self._emotion_predictions = EmotionPredictionRepository(session)
        self._incongruence_signals = IncongruenceSignalRepository(session)

    async def get_insights(
        self, *, conversation_id: uuid.UUID, user_id: uuid.UUID
    ) -> ConversationInsights:
        await self._conversations.get_owned(conversation_id=conversation_id, user_id=user_id)

        messages = await self._messages.list_for_conversation(conversation_id)
        timeline: list[TimelineEntry] = []
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        intent_counts: dict[str, int] = {}
        emotion_counts: dict[str, int] = {}
        mismatch_scores: list[float] = []
        high_mismatch_explanations: list[str] = []

        for message in messages:
            annotation = await self._nlp_annotations.get_for_message(message.id)
            if annotation is not None and annotation.sentiment_label in sentiment_counts:
                sentiment_counts[annotation.sentiment_label] += 1
            if annotation is not None:
                intent_counts[annotation.intent_label] = intent_counts.get(annotation.intent_label, 0) + 1

            emotion_entries: list[TimelineEmotionEntry] = []
            incongruence_entries: list[TimelineIncongruenceEntry] = []
            if message.audio_asset_id is not None:
                turns = await self._transcripts.get_aligned_turns(message.id)
                for turn in turns:
                    predictions = await self._emotion_predictions.list_for_turn(turn.id)
                    if predictions:
                        top = predictions[0]
                        emotion_entries.append(
                            TimelineEmotionEntry(
                                speaker_label=turn.speaker_label,
                                predicted_label=top.predicted_label,
                                confidence=top.confidence,
                            )
                        )
                        emotion_counts[top.predicted_label] = emotion_counts.get(top.predicted_label, 0) + 1
                signals = await self._incongruence_signals.list_for_message(message.id)
                for signal in signals:
                    incongruence_entries.append(
                        TimelineIncongruenceEntry(
                            incongruence_score=signal.incongruence_score,
                            confidence=signal.confidence,
                            explanation=signal.explanation,
                        )
                    )
                    mismatch_scores.append(signal.incongruence_score)
                    if signal.incongruence_score >= HIGH_MISMATCH_THRESHOLD:
                        high_mismatch_explanations.append(signal.explanation)

            timeline.append(
                TimelineEntry(
                    message_id=message.id,
                    role=message.role,
                    content=message.content,
                    created_at=message.created_at,
                    sentiment_label=annotation.sentiment_label if annotation else None,
                    sentiment_score=annotation.sentiment_score if annotation else None,
                    intent_label=annotation.intent_label if annotation else None,
                    topics=annotation.topics if annotation else [],
                    emotion=emotion_entries,
                    incongruence=incongruence_entries,
                )
            )

        observations = self._build_observations(
            sentiment_counts=sentiment_counts,
            intent_counts=intent_counts,
            emotion_counts=emotion_counts,
            mismatch_scores=mismatch_scores,
        )
        interpretations = self._build_interpretations(
            sentiment_counts=sentiment_counts,
            mismatch_scores=mismatch_scores,
            high_mismatch_count=len(high_mismatch_explanations),
        )

        return ConversationInsights(
            conversation_id=conversation_id,
            message_count=len(messages),
            timeline=timeline,
            observations=observations,
            interpretations=interpretations,
        )

    def _build_observations(
        self,
        *,
        sentiment_counts: dict[str, int],
        intent_counts: dict[str, int],
        emotion_counts: dict[str, int],
        mismatch_scores: list[float],
    ) -> list[Observation]:
        observations: list[Observation] = []
        total_sentiment = sum(sentiment_counts.values())
        if total_sentiment:
            parts = ", ".join(f"{label}: {count}" for label, count in sentiment_counts.items() if count)
            observations.append(
                Observation(
                    text=f"Sentiment was classified across {total_sentiment} message(s) - {parts}.",
                    basis="cardiffnlp/twitter-roberta-base-sentiment-latest (real sentiment model)",
                )
            )
        if intent_counts:
            top_intent = max(intent_counts.items(), key=lambda kv: kv[1])
            observations.append(
                Observation(
                    text=f"The most common detected intent was '{top_intent[0]}' ({top_intent[1]} message(s)).",
                    basis="deterministic rule-based intent classifier",
                )
            )
        if emotion_counts:
            top_emotion = max(emotion_counts.items(), key=lambda kv: kv[1])
            total_emotion = sum(emotion_counts.values())
            observations.append(
                Observation(
                    text=(
                        f"Vocal emotion was classified as '{top_emotion[0]}' in {top_emotion[1]} of "
                        f"{total_emotion} analyzed speaker turn(s)."
                    ),
                    basis="the currently active, trained emotion classifier model",
                )
            )
        if mismatch_scores:
            observations.append(
                Observation(
                    text=(
                        f"{len(mismatch_scores)} semantic-vocal incongruence signal(s) were computed, "
                        f"averaging {statistics.fmean(mismatch_scores):.2f} on a 0-1 scale."
                    ),
                    basis="deterministic comparison of real sentiment and real emotion signals - "
                    "an analytical signal, not a deception indicator",
                )
            )
        return observations

    def _build_interpretations(
        self,
        *,
        sentiment_counts: dict[str, int],
        mismatch_scores: list[float],
        high_mismatch_count: int,
    ) -> list[Interpretation]:
        interpretations: list[Interpretation] = []
        total_sentiment = sum(sentiment_counts.values())

        if total_sentiment >= 3 and sentiment_counts["negative"] / total_sentiment >= 0.5:
            interpretations.append(
                Interpretation(
                    text=(
                        "A majority of messages carried negative sentiment, which may suggest this "
                        "conversation touched on frustrating, difficult, or unresolved topics."
                    )
                )
            )
        if total_sentiment >= 3 and sentiment_counts["positive"] / total_sentiment >= 0.6:
            interpretations.append(
                Interpretation(
                    text=(
                        "Most messages carried positive sentiment, which may suggest the conversation "
                        "was generally constructive or well-received."
                    )
                )
            )
        if high_mismatch_count >= 2:
            interpretations.append(
                Interpretation(
                    text=(
                        f"{high_mismatch_count} turn(s) showed a notable divergence between what was said "
                        "and how it was said, which can occur with sarcasm, stress, fatigue, or suppressed "
                        "emotion - among many other real explanations."
                    )
                )
            )
        return interpretations
