"""`IncongruenceAnalyzer` implementation: a deterministic comparison of a
semantic signal (sentiment) and a vocal signal (emotion), both already
computed upstream by real models. This module performs no inference of its
own - it only combines two already-real signals - so there is nothing here
that could "fabricate" a result; the worst case is a low-confidence score
when upstream signals are themselves low-confidence, never an invented one.
"""
from __future__ import annotations

from voxmind.services.incongruence.interfaces import IncongruenceSignal
from voxmind.services.incongruence.valence import semantic_valence, vocal_valence


class DeterministicIncongruenceAnalyzer:
    async def analyze(self, semantic_signal: dict, vocal_signal: dict) -> IncongruenceSignal:
        sem_label = semantic_signal["sentiment_label"]
        sem_score = float(semantic_signal["sentiment_score"])
        voc_label = vocal_signal["predicted_label"]
        voc_confidence = float(vocal_signal["confidence"])

        sem_valence = semantic_valence(sem_label, sem_score)
        voc_valence = vocal_valence(voc_label, voc_confidence)

        # Max possible |difference| is 2.0 (fully-confident opposite poles);
        # normalized to a 0.0-1.0 incongruence score.
        raw_diff = abs(sem_valence - voc_valence)
        incongruence_score = min(raw_diff / 2.0, 1.0)

        # The measurement is only as trustworthy as its weaker input signal.
        confidence = min(sem_score, voc_confidence)

        if incongruence_score >= 0.6:
            qualifier = "a notable divergence"
        elif incongruence_score >= 0.3:
            qualifier = "a mild divergence"
        else:
            qualifier = "general alignment"

        explanation = (
            f"The words carried {sem_label} sentiment (score {sem_score:.2f}) while the vocal "
            f"tone was classified as {voc_label} (confidence {voc_confidence:.2f}), showing "
            f"{qualifier} between what was said and how it was said. This is an analytical "
            "signal only - it does not indicate deception, dishonesty, or intent, and should be "
            "interpreted alongside context (e.g. sarcasm, humor, fatigue, or suppressed emotion)."
        )

        return IncongruenceSignal(
            incongruence_score=incongruence_score,
            semantic_signal=semantic_signal,
            vocal_signal=vocal_signal,
            confidence=confidence,
            explanation=explanation,
        )
