"""DOCX parser: paragraphs and tables in document order."""
from __future__ import annotations

from pathlib import Path

import docx as docx_lib
from docx.table import Table as _Table
from docx.text.paragraph import Paragraph as _Paragraph

from .models import Cell, ParsedDocument, ParseError, Section, TableBlock


def parse_docx(path: Path) -> ParsedDocument:
    parsed = ParsedDocument(kind="docx", filename=path.name)
    try:
        document = docx_lib.Document(str(path))
    except Exception as exc:
        raise ParseError(f"could not parse DOCX {path.name}: {exc}") from exc
    table_number = 0
    for index, item in enumerate(document.iter_inner_content()):
        if isinstance(item, _Paragraph):
            parsed.sections.append(Section(index=index, kind="paragraph", text=item.text))
        elif isinstance(item, _Table):
            table_number += 1
            parsed.sections.append(
                Section(index=index, kind="table", table=_table_block(item, table_number))
            )
    return parsed


def _table_block(table: _Table, number: int) -> TableBlock:
    rows = []
    for r, row in enumerate(table.rows, start=1):
        cells = tuple(
            Cell(value=cell.text, ref=f"r{r}c{c}") for c, cell in enumerate(row.cells, start=1)
        )
        rows.append(cells)
    headers = tuple(cell.value for cell in rows[0]) if rows else ()
    return TableBlock(source=f"table {number}", headers=headers, rows=tuple(rows))
