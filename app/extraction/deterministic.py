"""Deterministic (rule-based) extraction, tried before any LLM call.

Two generic mechanisms, no domain knowledge:
1. Labelled values — a field's display name, key, or quoted phrases from its
   description matched against `Label: value` / `Label = value` lines and
   label-cell-plus-adjacent-value patterns, with strict ambiguity rejection.
2. Table columns — when the schema marks a primary field and a sheet header row
   matches enough field labels, one entry is produced per data row. Candidate
   tables are scored on more than column count: valid non-empty data rows,
   uniqueness of primary values, share of rows that look like data rows, mapped
   fields, and primary match quality. Rows kept are real records: blank rows,
   summary rows (totals/subtotals/averages by exact word match), rows with an
   empty primary, label-only rows, and duplicate primary identifiers are
   dropped — first occurrence wins. Ties between candidate tables are treated
   as ambiguous and left to the LLM fallback rather than guessed.

Sheet relevance: in multi-sheet workbooks, document-level values are taken from
the sheet that provides the table (or, without a table, the sheets matching the
most schema labels) — and on a table sheet only from rows above the header, so
column labels, row data, and summary blocks are never read as document-level
values. Other sheets contribute only when they match at least two distinct
schema labels, and only for fields no eligible sheet resolved — conflicting
sources with no clear winner stay unresolved.

Label matching priority: exact field-name/key matches beat exact quoted
description labels, which beat token-subset matches. A source label that
multiple fields claim with equal strength is contested and left unresolved
rather than arbitrarily assigned; schema order never breaks ties.

Fields are only resolved when the evidence is clean; everything else stays
unresolved for the LLM fallback."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..parsing.models import ParsedDocument
from .responses import FieldResult
from .schema import ExtractionSchema
from .values import parse_scalar

_LABEL_VALUE_RE = re.compile(r"^\s*(?P<label>.+?)\s*[:=]\s*(?P<value>.+?)\s*$")
_QUOTE_RE = re.compile(r"(?<!\w)'([^']{2,80})'(?!\w)|(?<!\w)\"([^\"]{2,80})\"(?!\w)")
_EG_SPLIT_RE = re.compile(r"\be\.?g\.?\b|\bfor example\b|\bfor instance\b", re.IGNORECASE)
_DATE_TOKEN_RE = re.compile(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b")
_SUMMARY_PRIMARY_WORDS = {
    "total", "totals", "subtotal", "subtotals", "grand total", "summary",
    "average", "avg", "mean", "median",
}
_MAX_HEADER_SCAN_ROWS = 30
_ADJACENT_RIGHT = 3
_ADJACENT_DOWN = 2
_CONFIDENCE = {2: 0.9, 1: 0.85, 0: 0.75}


def norm_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def quoted_labels(description: str) -> list[str]:
    """Normalized quoted phrases from a field description, ignoring examples
    (text from the first 'e.g.' / 'for example' onward). Possessive
    apostrophes ("record's") do not open or close quoted phrases."""
    if not description:
        return []
    head = _EG_SPLIT_RE.split(description)[0]
    out = []
    for m in _QUOTE_RE.finditer(head):
        norm = norm_label(m.group(1) or m.group(2))
        if norm:
            out.append(norm)
    return out


@dataclass
class DeterministicResult:
    entries: list[dict[str, FieldResult]]
    doc_level: dict[str, FieldResult]
    mode: str
    warnings: list[str] = field(default_factory=list)
    mapped_columns: set[str] | None = None


@dataclass
class _Candidate:
    label: str
    value: Any
    citation: str
    quality: int
    page_numbers: list[int] | None = None
    source_locations: list[str] | None = None
    sheet: str | None = None


def deterministic_extract(schema: ExtractionSchema, document: ParsedDocument) -> DeterministicResult:
    """Attempt full deterministic extraction. Returns one entry (single mode)
    or one entry per table row (table mode); unresolved fields are null."""
    warnings: list[str] = []
    exact, phrases = _field_sources(schema)
    table = None
    entries: list[dict[str, FieldResult]] | None = None
    mapped: set[str] | None = None
    table_sheets: set[str] = set()
    header_rows: dict[str, int] = {}
    if schema.primary_key:
        table = _table_extract(schema, document, exact, phrases, warnings)
        if table is not None:
            entries, mapped, table_sheets, header_rows = table
    eligible, admissible = _sheet_pools(document, exact, table_sheets)
    doc_level, contested = _labelled_values(
        schema, document, exact, phrases, eligible, admissible, header_rows
    )
    if contested:
        warnings.append(
            f"{contested} labelled source(s) matched multiple fields equally; left unresolved"
        )
    if entries:
        _overlay_doc_level(entries, doc_level, mapped)
        unresolved = [k for k in schema.fields if all(e[k].value is None for e in entries)]
        if unresolved:
            shown = ", ".join(unresolved[:8]) + ("..." if len(unresolved) > 8 else "")
            warnings.append(
                f"table extraction left {len(unresolved)} field(s) unresolved: {shown}"
            )
        return DeterministicResult(
            entries=entries, doc_level=doc_level, mode="table",
            warnings=warnings, mapped_columns=mapped,
        )
    if table is not None:
        warnings.append("table header matched but produced no data rows")
    elif schema.primary_key:
        warnings.append("no table header matched the schema strongly enough")
    entry = {k: doc_level.get(k) or FieldResult(value=None) for k in schema.fields}
    return DeterministicResult(entries=[entry], doc_level=doc_level, mode="single", warnings=warnings)


def _field_sources(schema: ExtractionSchema):
    """exact: normalized label -> [(field_key, quality)], keeping EVERY field that
    claims the label (quality 2 = name/key, 1 = quoted description) so equal-strength
    competition can be detected; phrases: all norms for token-subset matching."""
    exact: dict[str, list[tuple[str, int]]] = {}
    phrases: list[tuple[str, str]] = []
    for f in schema.fields.values():
        entries = [(norm_label(f.name), 2), (norm_label(f.key), 2)]
        entries += [(q, 1) for q in quoted_labels(f.description)]
        seen: set[str] = set()
        for norm, q in entries:
            if not norm or norm in seen:
                continue
            seen.add(norm)
            exact.setdefault(norm, []).append((f.key, q))
            phrases.append((norm, f.key))
    return exact, phrases


def _match_quality(
    label: str, exact: dict[str, list[tuple[str, int]]], phrases: list[tuple[str, str]]
) -> dict[str, int]:
    """field_key -> strongest match quality for this label: 2 = exact name/key
    match, 1 = exact quoted-description match, 0 = token-subset match."""
    out: dict[str, int] = {}

    def add(key: str, q: int):
        if key not in out or q > out[key]:
            out[key] = q

    for key, q in exact.get(label, ()):
        add(key, q)
    label_tokens = set(label.split())
    if label_tokens:
        for norm, key in phrases:
            phrase_tokens = set(norm.split())
            if not phrase_tokens:
                continue
            if (label_tokens <= phrase_tokens or phrase_tokens <= label_tokens) and max(
                len(label_tokens), len(phrase_tokens)
            ) >= 2:
                add(key, 0)
    return out


def _sheet_label_count(sheet, exact: dict[str, list[tuple[str, int]]]) -> int:
    """Number of distinct schema fields whose labels (exact name/key/quote
    matches) appear anywhere in the sheet — the sheet's relevance signal."""
    matched: set[str] = set()
    for row in sheet.rows:
        for cell in row:
            if cell.value is None or str(cell.value).strip() == "":
                continue
            for key, _q in exact.get(norm_label(cell.value), ()):
                matched.add(key)
    return len(matched)


def _sheet_pools(
    document: ParsedDocument, exact: dict[str, list[tuple[str, int]]], table_sheets: set[str]
) -> tuple[set[str], set[str]]:
    """(eligible, admissible) sheet names for document-level labelled values.

    Eligible: the sheet(s) that provide the table, or — without a table — every
    sheet matching the most schema labels. Admissible: remaining sheets matching
    at least two distinct schema labels; they may only fill fields that no
    eligible sheet resolved."""
    if not document.sheets:
        return set(), set()
    counts = {sheet.name: _sheet_label_count(sheet, exact) for sheet in document.sheets}
    if table_sheets:
        eligible = set(table_sheets)
    else:
        best = max(counts.values())
        eligible = {name for name, c in counts.items() if c == best} if best > 0 else set(counts)
    admissible = {name for name, c in counts.items() if c >= 2} - eligible
    return eligible, admissible


def _labelled_values(
    schema: ExtractionSchema,
    document: ParsedDocument,
    exact: dict[str, list[tuple[str, int]]],
    phrases: list[tuple[str, str]],
    eligible: set[str],
    admissible: set[str],
    header_rows: dict[str, int] | None = None,
) -> tuple[dict[str, FieldResult], int]:
    """Resolve document-level labelled values. A candidate whose label is claimed
    by several fields at equal strength is contested: it is assigned to no field,
    regardless of schema order. On a table sheet, only rows above the header are
    candidate sources. Returns (resolved fields, contested count)."""
    candidates = _collect_candidates(document, exact, eligible, admissible, header_rows or {})
    per_field: dict[str, list[_Candidate]] = {}
    contested = 0
    for cand in candidates:
        hits = _match_quality(cand.label, exact, phrases)
        if not hits:
            continue
        best_q = max(hits.values())
        top = [key for key, q in hits.items() if q == best_q]
        if len(top) > 1:
            contested += 1
            continue
        cand.quality = best_q
        per_field.setdefault(top[0], []).append(cand)
    resolved: dict[str, FieldResult] = {}
    for key, cands in per_field.items():
        pool = [c for c in cands if c.sheet is None or c.sheet in eligible]
        if not pool:
            pool = [c for c in cands if c.sheet is not None and c.sheet in admissible]
        if not pool:
            continue
        best_q = max(c.quality for c in pool)
        best = [c for c in pool if c.quality == best_q]
        values = {repr(c.value) for c in best}
        if len(best) > 1 and len(values) > 1:
            continue
        fr = _typed_doc_value(best[0])
        if fr is not None:
            resolved[key] = fr
    return resolved, contested


def _typed_doc_value(cand: _Candidate) -> FieldResult | None:
    """Conservative gate for document-level labelled values: clean numbers, dates,
    booleans, or digit-free strings resolve; anything mixed or range-like is left
    for the LLM (a plausible guess is worse than a null). Weak token-subset
    matches only resolve structured values, never free text."""
    parsed = parse_scalar(cand.value)
    if parsed is None:
        return None
    value, kind = parsed
    if kind == "string":
        if cand.quality == 0:
            return None
        if any(ch.isdigit() for ch in value):
            return None
        if len(_DATE_TOKEN_RE.findall(value)) >= 2:
            return None
    return FieldResult(
        value=value,
        citation=cand.citation,
        confidence=_CONFIDENCE[cand.quality],
        page_numbers=cand.page_numbers,
        source_locations=cand.source_locations,
    )


def _collect_candidates(
    document: ParsedDocument,
    exact: dict[str, list[tuple[str, int]]],
    eligible: set[str],
    admissible: set[str],
    header_rows: dict[str, int],
) -> list[_Candidate]:
    out: list[_Candidate] = []
    if document.kind == "pdf":
        for page in document.pages:
            for line in page.text.splitlines():
                _text_line_candidate(line, out, f"[Page {page.number}]", [page.number], None)
    elif document.kind == "docx":
        for section in document.sections:
            if section.kind == "paragraph":
                _text_line_candidate(section.text, out, "[Document]", None, None)
            elif section.table is not None:
                _grid_candidates(section.table.rows, out, section.table.source, exact)
    else:
        for sheet in document.sheets:
            if sheet.name in eligible or sheet.name in admissible:
                _sheet_candidates(sheet, out, exact, sheet.name, header_rows.get(sheet.name))
    return out


def _text_line_candidate(line: str, out: list[_Candidate], prefix: str, page_numbers, source_locations):
    m = _LABEL_VALUE_RE.match(line or "")
    if not m:
        return
    label, value = m.group("label"), m.group("value")
    if not value.strip() or not norm_label(label):
        return
    out.append(
        _Candidate(
            label=norm_label(label),
            value=value.strip(),
            citation=f"{prefix} {line.strip()[:100]}",
            quality=2,
            page_numbers=page_numbers,
            source_locations=source_locations,
        )
    )


def _grid_candidates(rows, out: list[_Candidate], source: str, exact: dict[str, list[tuple[str, int]]]):
    for row in rows:
        header_like = (
            sum(
                1
                for cell in row
                if cell.value not in (None, "") and norm_label(cell.value) in exact
            )
            >= 2
        )
        if header_like:
            continue
        for i, cell in enumerate(row):
            _text_line_candidate(str(cell.value or ""), out, f"[{source}, {cell.ref}]", None,
                                 [f"{source}!{cell.ref}"] if cell.ref else None)
            if i and cell.value not in (None, ""):
                label_cell = row[0]
                if label_cell.value not in (None, ""):
                    nl = norm_label(label_cell.value)
                    if nl:
                        out.append(
                            _Candidate(
                                label=nl,
                                value=cell.value,
                                citation=f"[{source}, {cell.ref}] {_snippet(cell.value)}",
                                quality=1,
                                source_locations=[f"{source}!{cell.ref}"] if cell.ref else None,
                            )
                        )


def _sheet_candidates(
    sheet,
    out: list[_Candidate],
    exact: dict[str, list[tuple[str, int]]],
    sheet_name: str,
    header_idx: int | None = None,
):
    """Collect label-value candidates from one sheet. When the sheet provides the
    table, only rows above the header are scanned: cells at or below the header
    are column labels, row data, or summary blocks — never document-level values."""
    rows = sheet.rows
    scan_limit = header_idx if header_idx is not None else len(rows)
    for r, row in enumerate(rows[:scan_limit]):
        for c, cell in enumerate(row):
            if cell.value is None or str(cell.value).strip() == "":
                continue
            text = str(cell.value)
            m = _LABEL_VALUE_RE.match(text)
            if m and m.group("value").strip():
                out.append(
                    _Candidate(
                        label=norm_label(m.group("label")),
                        value=m.group("value").strip(),
                        citation=f"[Sheet: {sheet.name}, Cell: {cell.ref}] {_snippet(m.group('value'))}",
                        quality=2,
                        source_locations=[f"{sheet.name}!{cell.ref}"],
                        sheet=sheet_name,
                    )
                )
                continue
            label = norm_label(text)
            if not label or len(label) > 60 or len(label.split()) > 6:
                continue
            right = _right_nonempty(rows, r, c)
            down = _down_nonempty(rows, r, c)
            right = right if right is not None and norm_label(right[0].value) not in exact else None
            down = down if down is not None and norm_label(down[0].value) not in exact else None
            chosen = _prefer_value(rows, right, down)
            if chosen is None:
                continue
            value_cell = chosen[0]
            out.append(
                _Candidate(
                    label=label,
                    value=value_cell.value,
                    citation=f"[Sheet: {sheet.name}, Cell: {value_cell.ref}] {_snippet(value_cell.value)}",
                    quality=1,
                    source_locations=[f"{sheet.name}!{value_cell.ref}"],
                    sheet=sheet_name,
                )
            )


def _right_nonempty(rows, r, c):
    row = rows[r]
    for j in range(c + 1, min(len(row), c + 1 + _ADJACENT_RIGHT)):
        if row[j].value not in (None, ""):
            return (row[j], r, j)
    return None


def _down_nonempty(rows, r, c):
    for i in range(r + 1, min(len(rows), r + 1 + _ADJACENT_DOWN)):
        row = rows[i]
        if c < len(row) and row[c].value not in (None, ""):
            return (row[c], i, c)
    return None


_KIND_RANK = {"number": 0, "date": 1, "bool": 2, "string": 3}


def _prefer_value(rows, right, down):
    """When a label cell has both a right and a below neighbour, numbers/dates
    beat text and non-header-like text beats header-like text; ties go to the
    right neighbour (the classic label: value layout). Each candidate is a
    (cell, row_index, col_index) tuple."""
    if right is None:
        return down
    if down is None:
        return right
    (cell_r, _r, _c), (cell_d, _r2, _c2) = right, down
    kind_r = parse_scalar(cell_r.value)
    kind_d = parse_scalar(cell_d.value)
    rank_r = _KIND_RANK[kind_r[1]] if kind_r else 3
    rank_d = _KIND_RANK[kind_d[1]] if kind_d else 3
    if rank_r != rank_d:
        return right if rank_r < rank_d else down
    header_r = _header_like(rows, right)
    header_d = _header_like(rows, down)
    if header_r != header_d:
        return down if header_r else right
    return right


def _header_like(rows, candidate) -> bool:
    """A digit-free text cell with a non-empty cell directly below it looks like
    a column header rather than a value."""
    cell, i, j = candidate
    if cell is None or any(ch.isdigit() for ch in str(cell.value)):
        return False
    if i + 1 >= len(rows) or j >= len(rows[i + 1]):
        return False
    return rows[i + 1][j].value not in (None, "")


def _match_header_row(sheet, schema: ExtractionSchema, exact, phrases):
    """Find the row that best matches the schema labels; return (row_index, colmap,
    matched_field_count) where colmap maps field key -> (column_index, quality).

    A column claimed by several fields at equal strength is contested and left
    unclaimed (but still counts as matched for header detection); a field matching
    several columns at equal strength is dropped. Schema order never breaks ties."""
    best: tuple[int, int, dict[str, tuple[int, int]], int] | None = None
    for r in range(min(len(sheet.rows), _MAX_HEADER_SCAN_ROWS)):
        row = sheet.rows[r]
        col_hits: dict[int, dict[str, int]] = {}
        for c, cell in enumerate(row):
            if cell.value is None or str(cell.value).strip() == "":
                continue
            label = norm_label(cell.value)
            if not label:
                continue
            for key, quality in _match_quality(label, exact, phrases).items():
                fmap = col_hits.setdefault(c, {})
                if key not in fmap or quality > fmap[key]:
                    fmap[key] = quality
        if not col_hits:
            continue
        field_cols: dict[str, list[tuple[int, int]]] = {}
        matched_fields: set[str] = set()
        for col, fmap in col_hits.items():
            best_q = max(fmap.values())
            matched_fields.update(fmap)
            top = [key for key, q in fmap.items() if q == best_q]
            if len(top) > 1:
                continue
            field_cols.setdefault(top[0], []).append((col, best_q))
        colmap: dict[str, tuple[int, int]] = {}
        for key, options in field_cols.items():
            best_q = max(q for _col, q in options)
            top_cols = [col for col, q in options if q == best_q]
            if len(top_cols) > 1:
                continue
            colmap[key] = (top_cols[0], best_q)
        if len(matched_fields) >= 2 and (best is None or len(matched_fields) > best[3]):
            best = (len(colmap), r, colmap, len(matched_fields))
    if best is None:
        return None
    return best[1], best[2], best[3]


@dataclass
class _TableCandidate:
    """One (sheet, header row) table candidate with quality signals."""

    sheet: Any
    header_idx: int
    colmap: dict
    matched_fields: int
    entries: list
    rows_after_header: int
    duplicates_dropped: int
    non_record_rows_dropped: int
    primary_quality: int

    @property
    def valid_rows(self) -> int:
        return len(self.entries)

    def sort_key(self) -> tuple:
        """Higher is better: valid data rows, unique primary values, share of
        rows that look like data rows, mapped fields, primary match quality,
        header match breadth."""
        total = self.valid_rows + self.duplicates_dropped
        unique = self.valid_rows / total if total else 0.0
        ratio = self.valid_rows / self.rows_after_header if self.rows_after_header else 0.0
        return (
            self.valid_rows,
            round(unique, 4),
            round(ratio, 4),
            len(self.colmap),
            self.primary_quality,
            self.matched_fields,
        )


def _table_extract(schema, document, exact, phrases, warnings):
    """Returns (entries, mapped_keys, table_sheet_names) or None when no sheet
    header matches the schema with its primary field, when the best candidate
    has no data rows, or when the top candidates tie (ambiguous)."""
    candidates: list[_TableCandidate] = []
    for sheet in document.sheets:
        found = _match_header_row(sheet, schema, exact, phrases)
        if found is None:
            continue
        header_idx, colmap, matched = found
        if schema.primary_key not in colmap:
            continue
        entries, rows_after, duplicates, non_records = _build_entries(schema, sheet, header_idx, colmap)
        candidates.append(
            _TableCandidate(
                sheet=sheet,
                header_idx=header_idx,
                colmap=colmap,
                matched_fields=matched,
                entries=entries,
                rows_after_header=rows_after,
                duplicates_dropped=duplicates,
                non_record_rows_dropped=non_records,
                primary_quality=colmap[schema.primary_key][1],
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda c: c.sort_key(), reverse=True)
    if len(candidates) > 1 and candidates[1].sort_key() == candidates[0].sort_key():
        warnings.append("table candidates are ambiguous; leaving extraction to the LLM")
        return None
    top = candidates[0]
    if not top.entries:
        warnings.append("table header matched but produced no data rows")
        return None
    if top.duplicates_dropped:
        warnings.append(
            f"{top.duplicates_dropped} duplicate primary row(s) skipped (first occurrence kept)"
        )
    if top.non_record_rows_dropped:
        warnings.append(
            f"{top.non_record_rows_dropped} non-record row(s) skipped "
            "(no values in any numeric column of a numeric-heavy table)"
        )
    return top.entries, set(top.colmap), {top.sheet.name}, {top.sheet.name: top.header_idx}


def _build_entries(schema, sheet, header_idx: int, colmap: dict):
    """One entry per data row below the header. Skips blank rows, summary rows
    (exact word match on the primary value), label-only rows, duplicate
    primary identifiers (first wins), and — in numeric-heavy tables — rows
    carrying no numeric data at all (summary and narrative blocks).

    Block-structured tables: a row with an empty primary cell that follows a
    valid record row is a CONTINUATION row. Its values are attached to that
    record via (a) header-mapped columns (fill nulls only) and (b) category
    matching against quoted phrases in field descriptions; fields whose
    description requests a sum ("sum of ...") aggregate the unclaimed
    continuation rows. Continuation mapping is conservative: ambiguous cells
    are dropped, summary rows terminate blocks, and rows never attach across
    sheet/table boundaries."""
    pcol, _pq = colmap[schema.primary_key]
    value_cols = [col for key, (col, _q) in colmap.items() if key != schema.primary_key]
    category_labels: dict[str, set[str]] = {}
    aggregate_fields: set[str] = set()
    for key, f in schema.fields.items():
        labels = quoted_labels(f.description)
        if labels:
            category_labels[key] = set(labels)
        if re.search(r"\b(?:sum|total|combined)\s+of\b", f.description or "", re.IGNORECASE):
            aggregate_fields.add(key)
    seen_primary: set[str] = set()
    staged: list[tuple[Any, str]] = []
    blocks: list[tuple[dict[str, FieldResult], list]] = []
    current: tuple[dict[str, FieldResult], list] | None = None
    rows_after = 0
    duplicates = 0
    for row in sheet.rows[header_idx + 1 :]:
        rows_after += 1
        if all(cell.value in (None, "") for cell in row):
            continue
        primary_cell = row[pcol] if pcol < len(row) else None
        pnorm = norm_label(primary_cell.value) if primary_cell and primary_cell.value is not None else ""
        if not pnorm:
            if current is not None:
                current[1].append(row)
            continue
        if pnorm in _SUMMARY_PRIMARY_WORDS:
            current = None
            continue
        if pnorm in seen_primary:
            duplicates += 1
            current = None
            continue
        seen_primary.add(pnorm)
        record = _row_entry(schema, sheet, row, colmap)
        staged.append((row, pnorm))
        current = (record, [])
        blocks.append(current)
    for record, cont_rows in blocks:
        _apply_block_continuations(
            schema, sheet, record, cont_rows, colmap, category_labels, aggregate_fields
        )
    kept, non_records = _drop_non_records(staged, colmap, schema.primary_key)
    kept_ids = {id(row) for row, _pnorm in kept}
    entries = [
        blocks[i][0] for i in range(len(staged)) if id(staged[i][0]) in kept_ids
    ]
    return entries, rows_after, duplicates, non_records


def _category_matches(nval: str, labels: set[str]) -> bool:
    if nval in labels:
        return True
    nval_tokens = set(nval.split())
    return any(set(lab.split()) <= nval_tokens for lab in labels)


def _row_value_cell(row, after_col: int, numeric_only: bool) -> int | None:
    """Value cell for a continuation row: prefer the first numeric cell after
    the category cell; fall back to a single non-empty cell when exactly one
    remains (unambiguous)."""
    numeric: list[int] = []
    nonempty: list[int] = []
    for c in range(after_col + 1, len(row)):
        cell = row[c]
        if cell.value in (None, ""):
            continue
        nonempty.append(c)
        parsed = parse_scalar(cell.value)
        if parsed is not None and parsed[1] == "number":
            numeric.append(c)
    if numeric:
        return numeric[0] if len(numeric) == 1 else numeric[0]
    if numeric_only:
        return None
    return nonempty[0] if len(nonempty) == 1 else None


def _apply_block_continuations(
    schema,
    sheet,
    record: dict[str, FieldResult],
    cont_rows: list,
    colmap: dict,
    category_labels: dict[str, set[str]],
    aggregate_fields: set[str],
):
    """Attach continuation rows to their block record.

    Per continuation row:
    1. Column-mapped fill: a value under a header-mapped column fills that
       field's null on the record (unless the cell is a category label).
    2. Category matching: a cell whose text matches a quoted phrase from a
       field's description claims the row for that field; the value is the
       next scalar cell in the row (or an inline 'label: value'). A cell
       claimed by multiple fields, or a field matching multiple cells, is
       ambiguous and dropped rather than guessed.
    3. Aggregation: fields whose description requests a sum ("sum of ...")
       aggregate the numeric values of rows with no category claim, excluding
       summary rows and rows claimed by other fields."""
    per_row: list[tuple[dict[int, set[str]], set[int]]] = []
    for row in cont_rows:
        cell_fields: dict[int, set[str]] = {}
        for c, cell in enumerate(row):
            if cell.value in (None, ""):
                continue
            nval = norm_label(cell.value)
            if not nval:
                continue
            for key, labels in category_labels.items():
                if key in aggregate_fields:
                    continue
                if _category_matches(nval, labels):
                    cell_fields.setdefault(c, set()).add(key)
        contested = {c for c, keys in cell_fields.items() if len(keys) > 1}
        per_row.append((cell_fields, contested))

    claims: dict[str, tuple[int, Any]] = {}
    for row_i, (row, (cell_fields, contested)) in enumerate(zip(cont_rows, per_row)):
        for c, keys in sorted(cell_fields.items()):
            if c in contested:
                continue
            key = next(iter(keys))
            if key in claims:
                continue
            claims[key] = (c, row_i)

    claimed_rows = set()
    for key, (c, row_i) in claims.items():
        claimed_rows.add(row_i)

    # column-mapped fill per row (skip category cells and summary rows)
    for row_i, row in enumerate(cont_rows):
        cell_fields, contested = per_row[row_i]
        if contested:
            continue
        category_cols = set(claims[key][0] for key in claims if claims[key][1] == row_i)
        for key, (col, _q) in colmap.items():
            if key == schema.primary_key or col in category_cols:
                continue
            cell = row[col] if col < len(row) else None
            if cell is None or cell.value in (None, ""):
                continue
            if record[key].value is not None:
                continue
            parsed = parse_scalar(cell.value)
            if parsed is None or parsed[0] is None:
                continue
            record[key] = FieldResult(
                value=parsed[0],
                citation=f"[Sheet: {sheet.name}, Cell: {cell.ref}] {_snippet(cell.value)}",
                confidence=0.8,
                source_locations=[f"{sheet.name}!{cell.ref}"],
            )

    # direct category-claimed values
    for key, (c, row_i) in claims.items():
        if record[key].value is not None:
            continue
        row = cont_rows[row_i]
        cell = row[c]
        inline = _LABEL_VALUE_RE.match(str(cell.value))
        value = None
        value_cell = None
        if inline and inline.group("value").strip():
            value_cell = cell
            parsed = parse_scalar(inline.group("value"))
            value = parsed[0] if parsed else None
        else:
            vc = _row_value_cell(row, c, numeric_only=False)
            if vc is not None:
                value_cell = row[vc]
                parsed = parse_scalar(value_cell.value)
                value = parsed[0] if parsed else None
        if value_cell is None or value is None:
            continue
        record[key] = FieldResult(
            value=value,
            citation=f"[Sheet: {sheet.name}, Cell: {value_cell.ref}] {_snippet(value_cell.value)}",
            confidence=0.85,
            source_locations=[f"{sheet.name}!{value_cell.ref}"],
        )

    # aggregation over unclaimed, non-summary rows
    for key in aggregate_fields:
        if record[key].value is not None:
            continue
        own_labels = category_labels.get(key, set())
        values: list[tuple[Any, Any]] = []
        for row_i, row in enumerate(cont_rows):
            if row_i in claimed_rows:
                continue
            numeric = []
            for c, cell in enumerate(row):
                if cell.value in (None, ""):
                    continue
                nval = norm_label(cell.value)
                if nval in _SUMMARY_PRIMARY_WORDS:
                    continue
                if own_labels and _category_matches(nval, own_labels):
                    continue
                p = parse_scalar(cell.value)
                if p is not None and p[1] == "number":
                    numeric.append((c, cell, p[0]))
            if len(numeric) != 1:
                continue
            _c, cell, v = numeric[0]
            values.append((v, cell.ref))
        if not values:
            continue
        total = sum(v for v, _ref in values if isinstance(v, (int, float)))
        refs = [ref for _v, ref in values]
        record[key] = FieldResult(
            value=total,
            citation=(
                f"[Sheet: {sheet.name}, Cells: {', '.join(refs[:3])}"
                f"{'…' if len(refs) > 3 else ''}] sum of {len(values)} matching lines"
            ),
            confidence=0.8,
            source_locations=[f"{sheet.name}!{ref}" for ref in refs],
        )


def _drop_non_records(staged, colmap, primary_key):
    """In tables where most records carry numbers, rows with no numeric value in
    any numeric-bearing column are summary/narrative blocks, not records."""
    if len(staged) < 4:
        return staged, 0
    value_cols = [col for key, (col, _q) in colmap.items() if key != primary_key]
    numeric_cols = []
    for col in value_cols:
        values = [
            row[col].value
            for row, _pnorm in staged
            if col < len(row) and row[col].value not in (None, "")
        ]
        if len(values) < 3:
            continue
        numeric = sum(1 for v in values if _is_number(v))
        if numeric / len(values) >= 0.6:
            numeric_cols.append(col)
    if not numeric_cols:
        return staged, 0

    def has_number(row) -> bool:
        return any(col < len(row) and _is_number(row[col].value) for col in numeric_cols)

    with_numeric = sum(1 for row, _pnorm in staged if has_number(row))
    if with_numeric / len(staged) < 0.6:
        return staged, 0
    kept = [(row, pnorm) for row, pnorm in staged if has_number(row)]
    return kept, len(staged) - len(kept)


def _is_number(value) -> bool:
    parsed = parse_scalar(value)
    return parsed is not None and parsed[1] == "number"


def _row_entry(schema, sheet, row, colmap) -> dict[str, FieldResult]:
    entry: dict[str, FieldResult] = {}
    for key in schema.fields:
        if key not in colmap:
            entry[key] = FieldResult(value=None)
            continue
        col, quality = colmap[key]
        cell = row[col] if col < len(row) else None
        if cell is None or cell.value in (None, ""):
            entry[key] = FieldResult(value=None)
            continue
        parsed = parse_scalar(cell.value)
        value = parsed[0] if parsed else None
        entry[key] = FieldResult(
            value=value,
            citation=f"[Sheet: {sheet.name}, Cell: {cell.ref}] {_snippet(cell.value)}",
            confidence=_CONFIDENCE[quality],
            source_locations=[f"{sheet.name}!{cell.ref}"],
        )
    return entry


def _overlay_doc_level(entries, doc_level, mapped):
    for key, fr in doc_level.items():
        if key in mapped:
            continue
        if all(e[key].value is None for e in entries):
            for e in entries:
                e[key] = fr


def _snippet(value) -> str:
    return " ".join(str(value).split())[:80]
