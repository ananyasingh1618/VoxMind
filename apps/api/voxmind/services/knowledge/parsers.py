"""Real document parsing - no placeholder text extraction. `.txt`/`.md` are
decoded directly; `.pdf` is parsed with `pypdf` (page-by-page real text
extraction, not OCR - a scanned/image-only PDF will genuinely yield little
or no text, and that is surfaced honestly as a short/empty document rather
than papered over).
"""
from __future__ import annotations

import io

from pypdf import PdfReader

from voxmind.core.exceptions import UnsupportedDocumentFormatError

SUPPORTED_CONTENT_TYPES = {
    "text/plain": "txt",
    "text/markdown": "md",
    "application/pdf": "pdf",
}


def parse_document(*, content_type: str, filename: str, raw_bytes: bytes) -> str:
    kind = SUPPORTED_CONTENT_TYPES.get(content_type)
    if kind is None:
        # Fall back to the file extension - browsers/clients are inconsistent
        # about the Content-Type they send for .md files in particular.
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        kind = {"txt": "txt", "md": "md", "markdown": "md", "pdf": "pdf"}.get(ext)
    if kind is None:
        raise UnsupportedDocumentFormatError(
            f"Unsupported document type: {content_type!r} ({filename!r}). "
            "Supported: .txt, .md, .pdf."
        )

    if kind in ("txt", "md"):
        return raw_bytes.decode("utf-8", errors="replace")

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)
