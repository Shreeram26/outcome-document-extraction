"""Document ingestion layer.

Parsers are pure: no LLM, OCR or network calls. Each returns a ParsedDocument
carrying source locations (page numbers, cell references) for later citations.

    from app.parsing import parse_file, render_page

    parsed = parse_file("lease.pdf")          # ParsedDocument
    png = render_page("lease.pdf", page=3)    # PNG bytes for vision extraction
"""
from .csv import parse_csv
from .dispatcher import PARSERS, SUPPORTED_EXTENSIONS, parse_file
from .docx import parse_docx
from .models import (
    Cell,
    ParseError,
    ParsedDocument,
    Page,
    Section,
    Sheet,
    TableBlock,
    sheet_cell_source,
)
from .pdf import parse_pdf, render_page
from .xlsx import parse_xlsx

__all__ = [
    "PARSERS",
    "SUPPORTED_EXTENSIONS",
    "Cell",
    "Page",
    "ParseError",
    "ParsedDocument",
    "Section",
    "Sheet",
    "TableBlock",
    "parse_csv",
    "parse_docx",
    "parse_file",
    "parse_pdf",
    "parse_xlsx",
    "render_page",
    "sheet_cell_source",
]
