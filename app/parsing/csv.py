"""CSV parser: one sheet named 'Sheet1' (the convention used in the reference outputs),
with A1-style cell references. Delimiter is sniffed, falling back to comma."""
from __future__ import annotations

import csv
import io
from pathlib import Path

from openpyxl.utils import get_column_letter

from .models import Cell, ParsedDocument, ParseError, Sheet

CSV_SHEET_NAME = "Sheet1"


def parse_csv(path: Path) -> ParsedDocument:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ParseError(f"could not read {path.name}: {exc}") from exc
    text = raw.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text), _dialect(text)))
    parsed = ParsedDocument(kind="csv", filename=path.name)
    parsed.sheets.append(_sheet(rows))
    return parsed


def _dialect(text: str) -> any:
    try:
        return csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def _sheet(rows_raw: list[list[str]]) -> Sheet:
    rows: list[tuple[Cell, ...]] = []
    width = 0
    for r, raw in enumerate(rows_raw, start=1):
        cells = tuple(
            Cell(value=value, ref=f"{get_column_letter(c)}{r}")
            for c, value in enumerate(raw, start=1)
        )
        while cells and cells[-1].value == "":
            cells = cells[:-1]
        rows.append(cells)
        width = max(width, len(cells))
    return Sheet(name=CSV_SHEET_NAME, rows=tuple(rows), n_rows=len(rows), n_cols=width)
