from __future__ import annotations

from app.extraction import context_phrases, normalize_schema, select_context, render_document
from app.parsing.models import Cell, Page, ParsedDocument, Sheet

SCHEMA = normalize_schema(
    {
        "fields": {
            "zeta_count": {"name": "Zeta Count", "description": "From the 'Zeta' line."},
            "kappa_note": {"name": "Kappa Note", "description": "Any note about kappa."},
        }
    }
)


def _big_sheet_doc(n_rows=400):
    rows = [(Cell("Header", "A1"), Cell("Value", "B1"))]
    for i in range(2, n_rows):
        label = "Zeta" if i % 50 == 0 else f"filler row {i} padding padding padding"
        rows.append((Cell(label, f"A{i}"), Cell(i, f"B{i}")))
    sheet = Sheet(name="Big", rows=tuple(rows), n_rows=n_rows, n_cols=2)
    return ParsedDocument(kind="xlsx", filename="big.xlsx", sheets=[sheet])


def test_context_phrases_from_names_and_quotes():
    phrases = context_phrases(SCHEMA.fields.values())
    assert "zeta count" in phrases
    assert "zeta" in phrases
    assert "kappa note" in phrases


def test_selection_keeps_matched_rows_and_head_only():
    doc = _big_sheet_doc()
    assert len(render_document(doc)) > 20000
    ctx, note = select_context(doc, SCHEMA.fields.values(), max_chars=6000)
    assert ctx.kind == "xlsx"
    assert len(ctx.sheets) == 1
    assert ctx.sheets[0].name == "Big"
    assert len(ctx.sheets[0].rows) < 30
    labels = [row[0].value for row in ctx.sheets[0].rows]
    assert "Header" in labels
    assert "Zeta" in labels
    assert note is not None and "trimmed" in note


def test_selection_budget_respected():
    doc = _big_sheet_doc(2000)
    ctx, _ = select_context(doc, SCHEMA.fields.values(), max_chars=4000)
    assert len(render_document(ctx)) <= 4200


def test_pdf_pages_selected_by_score():
    doc = ParsedDocument(
        kind="pdf",
        filename="d.pdf",
        pages=[
            Page(number=1, text="Title page with a Kappa mention"),
            Page(number=2, text="irrelevant padding " * 50),
            Page(number=3, text="Zeta Count equals twelve"),
            Page(number=4, text="more irrelevant padding " * 50),
        ],
    )
    ctx, _ = select_context(doc, SCHEMA.fields.values(), max_chars=400)
    numbers = [p.number for p in ctx.pages]
    assert 1 in numbers
    assert 3 in numbers
    assert 4 not in numbers


def test_head_block_always_included_even_without_match():
    doc = ParsedDocument(
        kind="pdf",
        filename="d.pdf",
        pages=[Page(number=1, text="Document title"), Page(number=2, text="nothing relevant here")],
    )
    schema = normalize_schema({"fields": {"x": {"name": "Omega", "description": ""}}})
    ctx, _ = select_context(doc, schema.fields.values(), max_chars=10000)
    assert ctx.pages[0].text == "Document title"
