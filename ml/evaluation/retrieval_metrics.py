"""Standard information-retrieval metrics computed over a real ranked list
of (item, is_relevant) pairs and a ground-truth relevance set. Pure
functions, no dependency on the app or the database - the caller
(ml/evaluation/evaluate_retrieval.py) is responsible for producing the
ranked list from a real retrieval run and a real relevance judgment file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def recall_at_k(ranked_relevant: list[bool], total_relevant: int, k: int) -> float | None:
    if total_relevant == 0:
        return None
    hits = sum(1 for r in ranked_relevant[:k] if r)
    return hits / total_relevant


def precision_at_k(ranked_relevant: list[bool], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for r in ranked_relevant[:k] if r)
    return hits / k


def reciprocal_rank(ranked_relevant: list[bool]) -> float:
    for i, is_relevant in enumerate(ranked_relevant, start=1):
        if is_relevant:
            return 1.0 / i
    return 0.0


def _drop_none(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def dcg_at_k(ranked_relevant: list[bool], k: int) -> float:
    return sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked_relevant[:k], start=1) if r)


def ndcg_at_k(ranked_relevant: list[bool], total_relevant: int, k: int) -> float | None:
    if total_relevant == 0:
        return None
    ideal = dcg_at_k([True] * min(total_relevant, k), k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_relevant, k) / ideal


@dataclass
class QueryResult:
    query: str
    ranked_relevant: list[bool]
    total_relevant: int


@dataclass
class RetrievalMetrics:
    k_values: list[int]
    n_queries: int
    mrr: float
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    per_query: list[dict]

    def to_dict(self) -> dict:
        return {
            "k_values": self.k_values,
            "n_queries": self.n_queries,
            "mrr": self.mrr,
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "precision_at_k": {str(k): v for k, v in self.precision_at_k.items()},
            "ndcg_at_k": {str(k): v for k, v in self.ndcg_at_k.items()},
            "per_query": self.per_query,
        }


def compute_retrieval_metrics(results: list[QueryResult], k_values: list[int]) -> RetrievalMetrics:
    if not results:
        raise ValueError("compute_retrieval_metrics requires at least one query result.")

    mrr_values = [reciprocal_rank(r.ranked_relevant) for r in results]
    recall: dict[int, float] = {}
    precision: dict[int, float] = {}
    ndcg: dict[int, float] = {}
    for k in k_values:
        recalls = _drop_none([recall_at_k(r.ranked_relevant, r.total_relevant, k) for r in results])
        recall[k] = sum(recalls) / len(recalls) if recalls else 0.0
        precision[k] = sum(precision_at_k(r.ranked_relevant, k) for r in results) / len(results)
        ndcgs = _drop_none([ndcg_at_k(r.ranked_relevant, r.total_relevant, k) for r in results])
        ndcg[k] = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

    per_query = [
        {
            "query": r.query,
            "reciprocal_rank": reciprocal_rank(r.ranked_relevant),
            "total_relevant": r.total_relevant,
            "rank_of_first_relevant": (
                next((i for i, v in enumerate(r.ranked_relevant, start=1) if v), None)
            ),
        }
        for r in results
    ]

    return RetrievalMetrics(
        k_values=k_values,
        n_queries=len(results),
        mrr=sum(mrr_values) / len(mrr_values),
        recall_at_k=recall,
        precision_at_k=precision,
        ndcg_at_k=ndcg,
        per_query=per_query,
    )
