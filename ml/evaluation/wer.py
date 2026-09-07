"""Real word/character error rate via standard Levenshtein edit distance -
no external dependency, no approximation. Both reference and hypothesis are
normalized (lowercased, punctuation stripped, whitespace collapsed) before
comparison, a standard, documented STT-evaluation convention: Whisper's
raw output casing/punctuation choices are not the property under test here,
transcription accuracy of the actual words is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = _PUNCTUATION_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


@dataclass
class SttSampleResult:
    sample_id: str
    reference_text: str
    hypothesis_text: str
    wer: float
    cer: float
    latency_ms: int


@dataclass
class SttMetrics:
    n_samples: int
    avg_wer: float
    avg_cer: float
    avg_latency_ms: float
    per_sample: list[dict]

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "avg_wer": self.avg_wer,
            "avg_cer": self.avg_cer,
            "avg_latency_ms": self.avg_latency_ms,
            "per_sample": self.per_sample,
        }


def aggregate_stt_metrics(results: list[SttSampleResult]) -> SttMetrics:
    if not results:
        raise ValueError("aggregate_stt_metrics requires at least one sample result.")
    return SttMetrics(
        n_samples=len(results),
        avg_wer=sum(r.wer for r in results) / len(results),
        avg_cer=sum(r.cer for r in results) / len(results),
        avg_latency_ms=sum(r.latency_ms for r in results) / len(results),
        per_sample=[
            {
                "sample_id": r.sample_id,
                "reference_text": r.reference_text,
                "hypothesis_text": r.hypothesis_text,
                "wer": r.wer,
                "cer": r.cer,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ],
    )
