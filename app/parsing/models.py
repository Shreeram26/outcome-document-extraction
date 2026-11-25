"""Common data model for parsed documents across all supported formats.

Every parser produces a `ParsedDocument` carrying source locations that the
extraction layer can turn into citations: 1-based page numbers for PDFs and
`SheetName!A1`-style cell references for XLSX/CSV.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ParseError(Exception):
    """Raised when a document cannot be parsed."""


@dataclass(frozen=True)
class Cell:
    """One spreadsheet/table cell. `ref` is 'A1' style (XLSX/CSV/DOCX tables), None when a location is not meaningful."""

    value: Any = None
    ref: str | None = None


@dataclass(frozen=True)
class TableBlock:
    """A table found in the document flow."""

    source: str
    headers: tuple[str, ...] = ()
    rows: tuple[tuple[Cell, ...], ...] = ()


@dataclass(frozen=True)
class Page:
    """One PDF page, 1-based."""

    number: int
    text: str


@dataclass(frozen=True)
class Section:
    """One block of a DOCX document flow, in reading order."""

    index: int
    kind: str
    text: str = ""
    table: TableBlock | None = None


@dataclass(frozen=True)
class Sheet:
    """One worksheet (or a CSV file as a single sheet). Rows keep their positions; trailing empty cells are trimmed."""

    name: str
    rows: tuple[tuple[Cell, ...], ...] = ()
    n_rows: int = 0
    n_cols: int = 0


@dataclass
class ParsedDocument:
    """Unified parse result. Only the fields relevant to `kind` are populated."""

    kind: str
    filename: str = ""
    pages: list[Page] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    sheets: list[Sheet] = field(default_factory=list)
    scanned_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def sheet_cell_source(sheet_name: str, ref: str) -> str:
    """Citation-ready location for a cell, e.g. 'Report1!A13'."""
    return f"{sheet_name}!{ref}"
