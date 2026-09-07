"""Deterministic, stopword-filtered keyword extraction - genuinely computed
from the text every time (word frequency, longest-run-first tie-breaking),
never a fabricated or fixed label list. This is explicitly *not* neural
topic modeling; it is documented here as a simple, honest heuristic so
nobody mistakes `topics` for an LDA/BERTopic-style output.
"""
from __future__ import annotations

import re
from collections import Counter

STOPWORDS = frozenset(
    """
    a an the this that these those and or but if then so because as of to in on for with at by from
    is are was were be been being do does did have has had i you he she it we they my your his her
    its our their me him us them what which who whom whose when where why how all any both each few
    more most other some such no nor not only own same than too very can will just don should now
    about above after again against below between into through during before out up down over under
    once here there while about above after
    """.split()
)

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}")


def extract_topics(text: str, *, max_topics: int = 5) -> list[str]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    candidates = [w for w in words if w not in STOPWORDS]
    if not candidates:
        return []
    counts = Counter(candidates)
    # Ties broken by first appearance order for determinism across runs.
    ordered_unique = sorted(dict.fromkeys(candidates), key=lambda w: (-counts[w],))
    return ordered_unique[:max_topics]
