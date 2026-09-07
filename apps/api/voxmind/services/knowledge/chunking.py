"""Deterministic paragraph-aware chunking: paragraphs are greedily packed
into chunks up to `chunk_size_chars`; a paragraph longer than that on its
own is hard-split with `overlap_chars` of context carried into the next
chunk. Same input always produces the same chunks - no randomness, no
truncation silently dropping content.
"""
from __future__ import annotations

import re

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def chunk_text(text: str, *, chunk_size_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size_chars:
            flush()
            start = 0
            while start < len(paragraph):
                end = start + chunk_size_chars
                chunks.append(paragraph[start:end].strip())
                start = end - overlap_chars if end < len(paragraph) else end
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > chunk_size_chars:
            flush()
            current = paragraph
        else:
            current = candidate

    flush()
    return [c for c in chunks if c]
