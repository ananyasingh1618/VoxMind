"""A real, deterministic extractive summarizer - frequency-weighted sentence
scoring (each sentence scored by the sum of its non-stopword words' document
frequency, normalized by sentence length; top-N sentences kept in their
original order). This is the well-established "Luhn-style" extractive
summarization algorithm, not a placeholder - and it is honestly labeled as
such (`method="extractive_fallback"`) rather than presented as equivalent to
an LLM-generated summary. Used only when no LLM provider is configured or
reachable; see services/memory_service.py.
"""
from __future__ import annotations

import re
from collections import Counter

from voxmind.services.nlp.topic_extractor import STOPWORDS

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}")


def summarize_extractive(text: str, *, max_sentences: int = 3) -> str:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    all_words = [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in STOPWORDS]
    frequencies = Counter(all_words)
    if not frequencies:
        return " ".join(sentences[:max_sentences])

    def score(sentence: str) -> float:
        words = [w.lower() for w in _WORD_RE.findall(sentence) if w.lower() not in STOPWORDS]
        if not words:
            return 0.0
        return sum(frequencies[w] for w in words) / len(words)

    ranked_indices = sorted(range(len(sentences)), key=lambda i: score(sentences[i]), reverse=True)
    top_indices = sorted(ranked_indices[:max_sentences])
    return " ".join(sentences[i] for i in top_indices)
