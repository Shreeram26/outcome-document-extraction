"""PDF parser: page-wise text, scanned-page detection, page rendering for vision use."""
from __future__ import annotations

from pathlib import Path

import pymupdf

from .models import Page, ParsedDocument, ParseError

MIN_TEXT_CHARS = 32
DEFAULT_RENDER_DPI = 150


def parse_pdf(path: Path) -> ParsedDocument:
    """Extract text per page. Pages with less than MIN_TEXT_CHARS of text that also
    contain an image are reported in `scanned_pages` (candidates for image-based
    extraction); pages with no text and no image are reported as blank."""
    parsed = ParsedDocument(kind="pdf", filename=path.name)
    try:
        with pymupdf.open(path) as doc:
            for number, page in enumerate(doc, start=1):
                text = page.get_text("text")
                parsed.pages.append(Page(number=number, text=text))
                if len(text.strip()) < MIN_TEXT_CHARS:
                    if page.get_images():
                        parsed.scanned_pages.append(number)
                    else:
                        parsed.warnings.append(f"page {number} is blank (no text, no images)")
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise ParseError(f"could not parse PDF {path.name}: {exc}") from exc
    if parsed.scanned_pages:
        shown = parsed.scanned_pages[:10]
        suffix = "..." if len(parsed.scanned_pages) > len(shown) else ""
        parsed.warnings.append(
            f"{len(parsed.scanned_pages)}/{len(parsed.pages)} pages have no text layer "
            f"(pages {shown}{suffix}); use image-based extraction for these"
        )
    return parsed


def render_page(path: Path, page_number: int, dpi: int = DEFAULT_RENDER_DPI) -> bytes:
    """Render a 1-based PDF page to PNG bytes, for vision-model extraction of scanned pages."""
    try:
        with pymupdf.open(path) as doc:
            if not 1 <= page_number <= len(doc):
                raise ParseError(f"page {page_number} out of range for {path.name} (1-{len(doc)})")
            return doc[page_number - 1].get_pixmap(dpi=dpi).tobytes("png")
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise ParseError(f"could not render page {page_number} of {path.name}: {exc}") from exc
