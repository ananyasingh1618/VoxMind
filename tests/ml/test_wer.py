"""Deterministic fixtures only - these are unit tests of the edit-distance
math, not real-world STT evaluation results (those come from
ml/evaluation/evaluate_stt.py against ml/evaluation/stt_datasets/)."""
from __future__ import annotations

import pytest

from ml.evaluation.wer import (
    aggregate_stt_metrics,
    character_error_rate,
    normalize_text,
    word_error_rate,
    SttSampleResult,
)


def test_identical_text_has_zero_error():
    assert word_error_rate("the quick brown fox", "the quick brown fox") == 0.0
    assert character_error_rate("the quick brown fox", "the quick brown fox") == 0.0


def test_normalization_ignores_case_and_punctuation():
    assert word_error_rate("The Quick Brown Fox.", "the quick brown fox") == 0.0
    assert normalize_text("Hello, World!") == "hello world"


def test_one_substitution_out_of_four_words():
    # "brown" -> "black": 1 substitution / 4 reference words
    assert word_error_rate("the quick brown fox", "the quick black fox") == pytest.approx(0.25)


def test_one_deletion_out_of_four_words():
    assert word_error_rate("the quick brown fox", "the quick fox") == pytest.approx(0.25)


def test_one_insertion_out_of_four_words():
    assert word_error_rate("the quick brown fox", "the very quick brown fox") == pytest.approx(0.25)


def test_empty_reference_with_empty_hypothesis_is_zero_error():
    assert word_error_rate("", "") == 0.0


def test_empty_reference_with_nonempty_hypothesis_is_full_error():
    assert word_error_rate("", "hello") == 1.0


def test_character_error_rate_on_single_substitution():
    # "cat" -> "cot": 1 character substitution / 3 reference characters
    assert character_error_rate("cat", "cot") == pytest.approx(1 / 3)


def test_aggregate_stt_metrics_averages_across_samples():
    results = [
        SttSampleResult("s1", "the cat sat", "the cat sat", wer=0.0, cer=0.0, latency_ms=100),
        SttSampleResult("s2", "the cat sat", "the dog sat", wer=1 / 3, cer=character_error_rate("the cat sat", "the dog sat"), latency_ms=200),
    ]
    metrics = aggregate_stt_metrics(results)
    assert metrics.n_samples == 2
    assert metrics.avg_wer == pytest.approx((0.0 + 1 / 3) / 2)
    assert metrics.avg_latency_ms == pytest.approx(150.0)
    assert len(metrics.per_sample) == 2


def test_aggregate_stt_metrics_requires_at_least_one_result():
    with pytest.raises(ValueError):
        aggregate_stt_metrics([])
