from voxmind.services.guardrails.output_validation import validate_structured_output
from voxmind.services.llm.interfaces import LlmResponse


def _response(**overrides) -> LlmResponse:
    base = dict(answer="A real answer.", citations=[], confidence=0.8, evidence_summary="Used context.")
    base.update(overrides)
    return LlmResponse(**base)


def test_well_formed_grounded_response_is_valid():
    result = validate_structured_output(_response(), grounding_status="grounded")
    assert result.is_valid
    assert result.issues == []


def test_grounded_status_with_empty_answer_is_invalid():
    result = validate_structured_output(_response(answer=""), grounding_status="grounded")
    assert not result.is_valid
    assert any("empty" in issue for issue in result.issues)


def test_confidence_out_of_range_is_invalid():
    result = validate_structured_output(_response(confidence=1.5), grounding_status="ungrounded")
    assert not result.is_valid
    assert any("confidence" in issue for issue in result.issues)


def test_leaked_chain_of_thought_marker_is_invalid():
    result = validate_structured_output(
        _response(answer="<thinking>let me reason about this</thinking> The answer is X."),
        grounding_status="ungrounded",
    )
    assert not result.is_valid
    assert any("reasoning" in issue for issue in result.issues)


def test_both_answer_and_evidence_summary_empty_is_invalid():
    result = validate_structured_output(
        _response(answer="", evidence_summary=""), grounding_status="unavailable"
    )
    assert not result.is_valid
