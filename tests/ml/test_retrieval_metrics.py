"""Deterministic fixtures only - these are unit tests of the IR-metric math,
not real-world retrieval evaluation results (those come from
ml/evaluation/evaluate_retrieval.py against a real hand-labeled dataset)."""
from __future__ import annotations

import pytest

from ml.evaluation.retrieval_metrics import (
    QueryResult,
    compute_retrieval_metrics,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_counts_hits_within_k():
    ranked = [False, True, False, True]
    assert recall_at_k(ranked, total_relevant=2, k=2) == pytest.approx(0.5)
    assert recall_at_k(ranked, total_relevant=2, k=4) == pytest.approx(1.0)


def test_recall_at_k_is_none_when_no_relevant_items_exist():
    assert recall_at_k([False, False], total_relevant=0, k=2) is None


def test_precision_at_k_divides_hits_by_k():
    ranked = [True, False, True, False]
    assert precision_at_k(ranked, k=2) == pytest.approx(0.5)
    assert precision_at_k(ranked, k=4) == pytest.approx(0.5)


def test_reciprocal_rank_finds_first_relevant_position():
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([True, False, False]) == pytest.approx(1.0)
    assert reciprocal_rank([False, False, False]) == 0.0


def test_dcg_rewards_earlier_relevant_ranks():
    early = dcg_at_k([True, False, False], k=3)
    late = dcg_at_k([False, False, True], k=3)
    assert early > late


def test_ndcg_is_one_for_ideal_ordering():
    assert ndcg_at_k([True, True, False], total_relevant=2, k=3) == pytest.approx(1.0)


def test_ndcg_is_none_when_no_relevant_items_exist():
    assert ndcg_at_k([False, False], total_relevant=0, k=2) is None


def test_compute_retrieval_metrics_over_two_queries():
    results = [
        QueryResult(query="q1", ranked_relevant=[True, False, False], total_relevant=1),
        QueryResult(query="q2", ranked_relevant=[False, True, False], total_relevant=1),
    ]
    metrics = compute_retrieval_metrics(results, k_values=[1, 3])

    assert metrics.n_queries == 2
    assert metrics.mrr == pytest.approx((1.0 + 0.5) / 2)
    assert metrics.recall_at_k[1] == pytest.approx(0.5)  # q1 hits@1, q2 misses@1
    assert metrics.recall_at_k[3] == pytest.approx(1.0)
    assert metrics.precision_at_k[1] == pytest.approx(0.5)
    assert len(metrics.per_query) == 2


def test_compute_retrieval_metrics_requires_at_least_one_query():
    with pytest.raises(ValueError):
        compute_retrieval_metrics([], k_values=[1])
