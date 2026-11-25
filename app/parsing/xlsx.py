"""XLSX parser: every sheet with cell values and A1-style references."""
from __future__ import annotations

from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from .models import Cell, ParsedDocument, ParseError, Sheet


def parse_xlsx(path: Path) -> ParsedDocument:
    """Reads cached values (`data_only=True`), so formula cells yield their last
    computed result rather than the formula string."""
    parsed = ParsedDocument(kind="xlsx", filename=path.name)
    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as exc:
        raise ParseError(f"could not parse XLSX {path.name}: {exc}") from exc
    try:
        for ws in wb.worksheets:
            parsed.sheets.append(_sheet(ws))
    finally:
        wb.close()
    return parsed


def _sheet(ws: Worksheet) -> Sheet:
    rows: list[tuple[Cell, ...]] = []
    width = 0
    for r, raw in enumerate(ws.iter_rows(values_only=True), start=1):
        cells = tuple(
            Cell(value=value, ref=f"{get_column_letter(c)}{r}")
            for c, value in enumerate(raw, start=1)
        )
        while cells and cells[-1].value in (None, ""):
            cells = cells[:-1]
        rows.append(cells)
        width = max(width, len(cells))
    return Sheet(name=ws.title, rows=tuple(rows), n_rows=len(rows), n_cols=width)
