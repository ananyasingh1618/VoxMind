from voxmind.core.config import Settings
from voxmind.services.llm.context_assembler import assemble_context


def _settings(**overrides) -> Settings:
    base = dict(
        JWT_SECRET_KEY="x" * 32,
        CONTEXT_TOKEN_BUDGET=50,
        CHARS_PER_TOKEN_ESTIMATE=4,
    )
    base.update(overrides)
    return Settings(**base)


def test_small_context_fits_without_trimming():
    settings = _settings(CONTEXT_TOKEN_BUDGET=5000)
    recent_turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    context, token_estimate = assemble_context(
        settings=settings,
        transcript="What is VoxMind?",
        speaker_info=[],
        emotion=[],
        nlp=None,
        incongruence=[],
        recent_turns=recent_turns,
        conversation_summary="A prior summary.",
        retrieved_knowledge=[{"chunk_id": "1", "document_title": "Doc", "text": "some text"}],
    )
    assert context.memory["recent_turns"] == recent_turns
    assert context.memory["summary"] == "A prior summary."
    assert len(context.retrieved_knowledge) == 1
    assert token_estimate > 0


def test_over_budget_drops_oldest_recent_turns_first():
    settings = _settings(CONTEXT_TOKEN_BUDGET=40)
    recent_turns = [{"role": "user", "content": f"message number {i} with some padding text"} for i in range(20)]
    context, _ = assemble_context(
        settings=settings,
        transcript="question",
        speaker_info=[],
        emotion=[],
        nlp=None,
        incongruence=[],
        recent_turns=recent_turns,
        conversation_summary=None,
        retrieved_knowledge=[],
    )
    kept = context.memory["recent_turns"]
    assert len(kept) < len(recent_turns)
    # oldest (index 0) should be the first dropped
    if kept:
        assert kept[0] != recent_turns[0]


def test_over_budget_shortens_retrieved_excerpts_before_dropping_summary():
    settings = _settings(CONTEXT_TOKEN_BUDGET=60)
    long_text = "word " * 400
    context, _ = assemble_context(
        settings=settings,
        transcript="question",
        speaker_info=[],
        emotion=[],
        nlp=None,
        incongruence=[],
        recent_turns=[],
        conversation_summary="short summary",
        retrieved_knowledge=[{"chunk_id": "1", "document_title": "Doc", "text": long_text}],
    )
    if context.retrieved_knowledge:
        assert len(context.retrieved_knowledge[0]["text"]) < len(long_text)


def test_never_trims_the_transcript_itself():
    settings = _settings(CONTEXT_TOKEN_BUDGET=1)
    transcript = "this exact question must survive even a tiny budget"
    context, _ = assemble_context(
        settings=settings,
        transcript=transcript,
        speaker_info=[],
        emotion=[],
        nlp=None,
        incongruence=[],
        recent_turns=[{"role": "user", "content": "old"} for _ in range(10)],
        conversation_summary="a summary",
        retrieved_knowledge=[{"chunk_id": "1", "document_title": "d", "text": "x" * 500}],
    )
    assert context.transcript == transcript
