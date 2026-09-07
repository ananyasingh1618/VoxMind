import pytest

from voxmind.services.llm.interfaces import ConversationContext
from voxmind.services.llm.mock_provider import MockLlmProvider
from voxmind.services.llm.parsing import parse_structured_response


def test_parses_well_formed_tool_payload():
    data = {
        "answer": "The answer.",
        "citations": [{"chunk_id": "abc", "document_title": "Doc", "excerpt": "quote"}],
        "confidence": 0.8,
        "evidence_summary": "Used chunk abc.",
    }
    response = parse_structured_response(data)
    assert response.answer == "The answer."
    assert response.citations[0].chunk_id == "abc"
    assert response.confidence == 0.8


def test_missing_optional_fields_default_safely():
    response = parse_structured_response({"answer": "ok"})
    assert response.citations == []
    assert response.confidence == 0.0
    assert response.evidence_summary == ""


@pytest.mark.asyncio
async def test_mock_provider_only_cites_chunks_it_was_actually_given():
    provider = MockLlmProvider()
    context = ConversationContext(
        transcript="What does the doc say?",
        speaker_info=[],
        emotion=[],
        nlp=None,
        incongruence=[],
        memory=None,
        retrieved_knowledge=[{"chunk_id": "real-chunk-1", "document_title": "Doc", "text": "content"}],
        system_instructions="",
        safety_context={},
    )
    response = await provider.generate(context)
    for citation in response.citations:
        assert citation.chunk_id == "real-chunk-1"


@pytest.mark.asyncio
async def test_mock_provider_cites_nothing_when_no_knowledge_retrieved():
    provider = MockLlmProvider()
    context = ConversationContext(
        transcript="hello",
        speaker_info=[],
        emotion=[],
        nlp=None,
        incongruence=[],
        memory=None,
        retrieved_knowledge=[],
        system_instructions="",
        safety_context={},
    )
    response = await provider.generate(context)
    assert response.citations == []
