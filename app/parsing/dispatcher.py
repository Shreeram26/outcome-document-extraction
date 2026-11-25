"""Dispatcher: pick a parser by file extension."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .csv import parse_csv
from .docx import parse_docx
from .models import ParsedDocument, ParseError
from .pdf import parse_pdf
from .xlsx import parse_xlsx

PARSERS: dict[str, Callable[[Path], ParsedDocument]] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".xlsx": parse_xlsx,
    ".xlsm": parse_xlsx,
    ".csv": parse_csv,
}

SUPPORTED_EXTENSIONS = sorted(PARSERS)


def parse_file(path: str | Path) -> ParsedDocument:
    """Parse a document into a ParsedDocument. Raises ParseError for missing
    files and unsupported extensions; parsers raise ParseError for unreadable content."""
    p = Path(path)
    if not p.is_file():
        raise ParseError(f"file not found: {p}")
    parser = PARSERS.get(p.suffix.lower())
    if parser is None:
        raise ParseError(
            f"unsupported file type {p.suffix!r}; supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return parser(p)
