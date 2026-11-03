"""PDF -> per-page text."""
from __future__ import annotations

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PdfError(ValueError):
    pass


def extract_pages(data: bytes) -> list[dict]:
    """Return [{"page": 1, "text": "..."}, ...] (1-based page numbers)."""
    if not data.startswith(b"%PDF"):
        raise PdfError("File is not a valid PDF")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Empty-password PDFs are common; anything else we cannot read.
            if not reader.decrypt(""):
                raise PdfError("PDF is password protected")
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            pages.append({"page": i, "text": text})
    except PdfReadError as exc:
        raise PdfError(f"Could not read PDF: {exc}") from exc
    if not pages:
        raise PdfError("PDF has no pages")
    return pages
