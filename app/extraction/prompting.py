"""Dynamic prompt construction: system rules + rendered document + schema-driven field list.

Nothing here knows about any specific document domain; everything field-specific
comes from the ExtractionSchema."""
from __future__ import annotations

from ..parsing.models import ParsedDocument, TableBlock
from .schema import ExtractionSchema

MAX_DOCUMENT_CHARS = 200_000

SYSTEM_PROMPT = """You are a precise document extraction engine.

Rules:
1. Use only the document provided below. Never guess and never use outside knowledge.
2. If a value is not stated in the document, set "value" to null, "citation" to "" and "confidence" to 0. A wrong value is worse than a null.
3. Numbers are JSON numbers: no currency symbols, no thousands separators, no units, no quotes.
4. Dates are strings in YYYY-MM-DD format.
5. Percentages exactly as the document prints them (e.g. 91.67, never 0.9167) as JSON numbers.
6. Booleans are true/false.
7. "citation" is a short verbatim snippet from the document prefixed with its location, e.g. "[Page 3] Base Rent: $4,250.00" or "[Sheet1!F4] 4250". At most 60 characters. Never fabricate snippets; copy them from the document.
8. "confidence" is a number in [0.0, 1.0]: your own estimate that the value is correct.
9. If the document contains a table with one row per record — the schema marks one field as primary; its value identifies each row — return one entry per row. Otherwise return exactly one entry for the whole document.
10. Every entry must carry every listed field; fields that apply to the whole document repeat the same value in every entry.
11. Reply with ONLY a JSON object of the form {"entries": [...]} — no prose, no markdown fences."""


def build_messages(schema: ExtractionSchema, document: ParsedDocument) -> list[dict[str, str]]:
    """Build the [system, user] message pair for one extraction call."""
    field_lines = []
    for f in schema.fields.values():
        line = f'- "{f.key}" ({f.name}): {f.description or "(no description)"}'
        if f.primary:
            line += " [primary: its value identifies one row]"
        field_lines.append(line)
    header = f"DOCUMENT ({document.filename or 'unnamed'}):\n\n"
    user = (
        header
        + render_document(document)
        + "\n\nFIELDS TO EXTRACT:\n"
        + "\n".join(field_lines)
    )
    system = SYSTEM_PROMPT
    if schema.primary_key:
        system += (
            "\n12. RECORD COMPLETENESS: The context may contain many records"
            " (table rows, ledger lines, list items). Extract EVERY record"
            " present in the supplied context — do not stop early and do not"
            " skip any. Each entry must contain EVERY field listed below;"
            " set a field to null only when it is genuinely absent for that"
            " record in the supplied context. Copy identifiers (codes, ids,"
            " numbers) exactly. Keep citations at most 60 characters."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_document(document: ParsedDocument) -> str:
    """Render a ParsedDocument as compact text that keeps source locations
    (page markers, cell references) so citations can point back into the document."""
    parts: list[str] = []
    if document.kind == "pdf":
        for page in document.pages:
            text = page.text.strip()
            if text:
                parts.append(f"--- Page {page.number} ---\n{text}")
            else:
                parts.append(f"--- Page {page.number} --- [no text layer; image only]")
    elif document.kind == "docx":
        for section in document.sections:
            if section.kind == "paragraph":
                if section.text.strip():
                    parts.append(section.text)
            elif section.table is not None:
                parts.append(_render_table(section.table))
    else:
        for sheet in document.sheets:
            parts.append(f"--- Sheet: {sheet.name} ---\n{_render_grid(sheet)}")
    text = "\n\n".join(parts)
    if len(text) > MAX_DOCUMENT_CHARS:
        text = text[:MAX_DOCUMENT_CHARS] + f"\n\n[document truncated at {MAX_DOCUMENT_CHARS} characters]"
    return text


def _render_grid(sheet) -> str:
    lines = []
    for row in sheet.rows:
        cells = [f"{c.ref}={_one_line(c.value)}" for c in row if c.value not in (None, "")]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _render_table(table: TableBlock) -> str:
    lines = [f"[{table.source}]"]
    if table.headers:
        lines.append(" | ".join(_one_line(h) for h in table.headers))
    for row in table.rows:
        lines.append(" | ".join(_one_line(c.value) for c in row))
    return "\n".join(lines)


def _one_line(value) -> str:
    return " ".join(str(value).split())
