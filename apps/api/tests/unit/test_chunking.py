from voxmind.services.knowledge.chunking import chunk_text


def test_short_text_yields_single_chunk():
    text = "This is a short document.\n\nIt has two paragraphs."
    chunks = chunk_text(text, chunk_size_chars=1000, overlap_chars=50)
    assert len(chunks) == 1
    assert "short document" in chunks[0]
    assert "two paragraphs" in chunks[0]


def test_paragraphs_are_packed_until_size_limit():
    paragraphs = [f"Paragraph number {i} with some real content." for i in range(20)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, chunk_size_chars=200, overlap_chars=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200 + 50  # small slack for the packing boundary


def test_no_content_is_silently_dropped():
    paragraphs = [f"Paragraph {i}." for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, chunk_size_chars=30, overlap_chars=5)
    joined = " ".join(chunks)
    for i in range(10):
        assert f"Paragraph {i}." in joined


def test_oversized_single_paragraph_is_hard_split_with_overlap():
    long_paragraph = "word " * 500  # ~2500 chars, no paragraph breaks
    chunks = chunk_text(long_paragraph, chunk_size_chars=500, overlap_chars=50)
    assert len(chunks) > 1


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n\n  ", chunk_size_chars=100, overlap_chars=10) == []


def test_deterministic():
    text = "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph."
    assert chunk_text(text, chunk_size_chars=25, overlap_chars=5) == chunk_text(
        text, chunk_size_chars=25, overlap_chars=5
    )
