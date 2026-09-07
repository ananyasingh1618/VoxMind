import uuid
from types import SimpleNamespace

from voxmind.services.retrieval.hybrid import reciprocal_rank_fusion


def _chunk():
    return SimpleNamespace(id=uuid.uuid4())


def test_candidate_ranked_first_in_both_lists_wins():
    a, b, c = _chunk(), _chunk(), _chunk()
    vector_results = [(a, 0.9), (b, 0.5), (c, 0.1)]
    lexical_results = [(a, 5.0), (c, 3.0), (b, 1.0)]
    fused = reciprocal_rank_fusion(vector_results=vector_results, lexical_results=lexical_results, rrf_k=60)
    assert fused[0].chunk is a


def test_candidate_present_in_only_one_list_still_included():
    a, b = _chunk(), _chunk()
    vector_results = [(a, 0.9)]
    lexical_results = [(b, 5.0)]
    fused = reciprocal_rank_fusion(vector_results=vector_results, lexical_results=lexical_results, rrf_k=60)
    chunk_ids = {c.chunk.id for c in fused}
    assert a.id in chunk_ids and b.id in chunk_ids


def test_empty_inputs_yield_empty_fusion():
    assert reciprocal_rank_fusion(vector_results=[], lexical_results=[], rrf_k=60) == []


def test_scores_and_ranks_preserved_per_source():
    a = _chunk()
    fused = reciprocal_rank_fusion(vector_results=[(a, 0.75)], lexical_results=[(a, 2.5)], rrf_k=60)
    candidate = fused[0]
    assert candidate.vector_score == 0.75
    assert candidate.vector_rank == 1
    assert candidate.lexical_score == 2.5
    assert candidate.lexical_rank == 1
    assert candidate.hybrid_score == 2 * (1 / 61)
