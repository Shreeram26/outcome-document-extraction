"""Context selection for the LLM fallback: instead of rendering the whole
document, keep the head plus the blocks that mention the unresolved fields'
labels. Small documents pass through unchanged (selection only trims when a
document is actually large)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..parsing.models import Cell, Page, ParsedDocument, Section, Sheet
from .deterministic import norm_label, quoted_labels
from .prompting import _one_line, _render_table
from .schema import ExtractionField

DEFAULT_CONTEXT_MAX_CHARS = 36_000
_SHEET_HEAD_ROWS = 5
_ROW_NEIGHBOURHOOD = 1


@dataclass
class _Block:
    key: str
    text: str
    page: int | None = None
    section: Section | None = None
    sheet_index: int | None = None
    row_index: int | None = None


def context_phrases(fields: Iterable[ExtractionField]) -> list[str]:
    phrases: set[str] = set()
    for f in fields:
        for src in (f.name, f.key):
            norm = norm_label(src)
            if norm:
                phrases.add(norm)
        phrases.update(quoted_labels(f.description))
    return sorted(phrases, key=len, reverse=True)


def select_context(
    document: ParsedDocument, fields: Iterable[ExtractionField], max_chars: int = DEFAULT_CONTEXT_MAX_CHARS
) -> tuple[ParsedDocument, str | None]:
    """Return (context document, note). The context keeps the document head and
    the highest-scoring blocks for the given fields, within max_chars."""
    phrases = context_phrases(fields)
    blocks = _blocks(document)

    def score(block: _Block) -> int:
        return sum(1 for p in phrases if p in block.key)

    chosen: set[int] = set()
    if blocks:
        chosen.add(0)
        for i, block in enumerate(blocks):
            if score(block) > 0:
                chosen.add(i)
    sheet_rows: dict[int, set[int]] = {}
    for i in chosen:
        b = blocks[i]
        if b.sheet_index is not None and b.row_index is not None:
            rows = sheet_rows.setdefault(b.sheet_index, set())
            for offset in range(-_ROW_NEIGHBOURHOOD, _ROW_NEIGHBOURHOOD + 1):
                rows.add(b.row_index + offset)
    for i, b in enumerate(blocks):
        if b.sheet_index is not None and b.row_index in sheet_rows.get(b.sheet_index, set()):
            chosen.add(i)

    selected: list[_Block] = []
    used = 0
    for i, b in enumerate(blocks):
        if i in chosen and used + len(b.text) + 2 <= max_chars:
            selected.append(b)
            used += len(b.text) + 2
    total = sum(len(b.text) for b in blocks)
    note = None if used >= total else f"context trimmed to {used} of {total} characters"
    return _rebuild(document, selected), note


def _blocks(document: ParsedDocument) -> list[_Block]:
    out: list[_Block] = []
    if document.kind == "pdf":
        for page in document.pages:
            if page.text.strip():
                out.append(_Block(key=norm_label(page.text), text=page.text, page=page.number))
    elif document.kind == "docx":
        for section in document.sections:
            if section.kind == "paragraph" and section.text.strip():
                out.append(_Block(key=norm_label(section.text), text=section.text, section=section))
            elif section.table is not None:
                text = _render_table(section.table)
                out.append(_Block(key=norm_label(text), text=text, section=section))
    else:
        for idx, sheet in enumerate(document.sheets):
            for r, row in enumerate(sheet.rows):
                cells = [f"{c.ref}={_one_line(c.value)}" for c in row if c.value not in (None, "")]
                if not cells:
                    continue
                line = " | ".join(cells)
                out.append(_Block(key=norm_label(line), text=line, sheet_index=idx, row_index=r))
    return out


def _rebuild(document: ParsedDocument, selected: list[_Block]) -> ParsedDocument:
    if document.kind == "pdf":
        pages = [Page(number=b.page, text=b.text) for b in selected if b.page is not None]
        return ParsedDocument(kind="pdf", filename=document.filename, pages=pages)
    if document.kind == "docx":
        sections = [b.section for b in selected if b.section is not None]
        return ParsedDocument(kind="docx", filename=document.filename, sections=sections)
    by_sheet: dict[int, list[int]] = {}
    for b in selected:
        if b.sheet_index is not None and b.row_index is not None:
            by_sheet.setdefault(b.sheet_index, []).append(b.row_index)
    sheets = []
    for idx in sorted(by_sheet):
        original = document.sheets[idx]
        rows = tuple(original.rows[r] for r in sorted(set(by_sheet[idx])) if r < len(original.rows))
        sheets.append(Sheet(name=original.name, rows=rows, n_rows=len(rows), n_cols=original.n_cols))
    return ParsedDocument(kind=document.kind, filename=document.filename, sheets=sheets)
