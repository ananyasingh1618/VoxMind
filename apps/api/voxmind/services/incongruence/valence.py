"""Deterministic valence mappings used to compare a semantic signal
(sentiment) against a vocal signal (emotion) on the same numeric scale
(-1.0 = negative, 0.0 = neutral, +1.0 = positive). These are documented,
fixed mappings - not learned - so the incongruence score stays fully
explainable. `surprised` is deliberately mapped to neutral valence: surprise
is an arousal signal with ambiguous polarity (it can accompany good or bad
news), so treating it as positive or negative would misrepresent it.
"""
from __future__ import annotations

SENTIMENT_VALENCE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

EMOTION_VALENCE = {
    "happy": 1.0,
    "calm": 0.5,
    "neutral": 0.0,
    "surprised": 0.0,
    "sad": -1.0,
    "angry": -1.0,
    "fearful": -1.0,
    "disgust": -1.0,
}


def semantic_valence(sentiment_label: str, sentiment_score: float) -> float:
    return SENTIMENT_VALENCE.get(sentiment_label, 0.0) * sentiment_score


def vocal_valence(emotion_label: str, confidence: float) -> float:
    return EMOTION_VALENCE.get(emotion_label, 0.0) * confidence
