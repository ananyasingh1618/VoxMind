from voxmind.services.llm.grounding import validate_citations
from voxmind.services.llm.interfaces import Citation


def _citation(chunk_id: str) -> Citation:
    return Citation(chunk_id=chunk_id, document_title="Doc", excerpt="excerpt text")


def test_all_valid_citations_are_grounded():
    citations = [_citation("a"), _citation("b")]
    valid, status, details = validate_citations(citations, {"a", "b", "c"})
    assert status == "grounded"
    assert len(valid) == 2
    assert details["invalid_citation_ids"] == []


def test_fabricated_citation_id_is_filtered_out_and_flagged():
    citations = [_citation("a"), _citation("invented-id")]
    valid, status, details = validate_citations(citations, {"a", "b"})
    assert status == "partially_grounded"
    assert [c.chunk_id for c in valid] == ["a"]
    assert details["invalid_citation_ids"] == ["invented-id"]


def test_all_invalid_citations_yields_ungrounded_and_empty_valid_list():
    citations = [_citation("invented-1"), _citation("invented-2")]
    valid, status, details = validate_citations(citations, {"a"})
    assert status == "ungrounded"
    assert valid == []
    assert details["citations_valid"] == 0


def test_no_citations_returned_with_retrieved_context_is_ungrounded():
    valid, status, details = validate_citations([], {"a", "b"})
    assert status == "ungrounded"
    assert valid == []
    assert details["retrieved_chunk_count"] == 2


def test_empty_retrieval_is_always_ungrounded_regardless_of_citations():
    """Missing-source handling: if nothing was retrieved, no claim can be
    called grounded in a document no matter what the model claims to cite."""
    citations = [_citation("hallucinated")]
    valid, status, _details = validate_citations(citations, set())
    assert status == "ungrounded"
    assert valid == []
