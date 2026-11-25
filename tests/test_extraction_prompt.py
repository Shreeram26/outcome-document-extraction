from __future__ import annotations

from app.extraction import build_messages, normalize_schema, render_document
from app.parsing.models import Cell, Page, ParsedDocument, Section, Sheet, TableBlock

SCHEMA = normalize_schema(
    {
        "name": "widgets",
        "fields": {
            "widget_count": {"name": "Widget Count", "description": "How many widgets."},
            "made_on": {"name": "Made On", "description": "Manufacture date."},
            "serial": {"name": "Serial", "description": "Serial number.", "primary": True},
        },
    }
)


def _pdf_doc() -> ParsedDocument:
    doc = ParsedDocument(kind="pdf", filename="x.pdf")
    doc.pages = [Page(number=1, text="Hello world"), Page(number=2, text="")]
    return doc


def test_messages_shape():
    messages = build_messages(SCHEMA, _pdf_doc())
    assert [m["role"] for m in messages] == ["system", "user"]
    assert isinstance(messages[0]["content"], str) and messages[0]["content"]
    assert isinstance(messages[1]["content"], str) and messages[1]["content"]


def test_system_prompt_carries_the_core_rules():
    system = build_messages(SCHEMA, _pdf_doc())[0]["content"]
    for fragment in ("null", "YYYY-MM-DD", "confidence", "citation", '{"entries"'):
        assert fragment in system


def test_user_message_contains_every_field_dynamically():
    user = build_messages(SCHEMA, _pdf_doc())[1]["content"]
    for fragment in (
        '"widget_count"',
        "Widget Count",
        "How many widgets.",
        '"made_on"',
        '"serial"',
        "[primary",
        "x.pdf",
    ):
        assert fragment in user


def test_prompts_have_no_domain_hardcoding():
    for content, _ in build_messages(SCHEMA, _pdf_doc()):
        low = content.lower()
        for word in ("tenant", "landlord", "lease", "rent roll", "box score", "gl code", "unit type"):
            assert word not in low, f"domain-specific term leaked into prompt: {word}"


def test_render_pdf_pages_with_markers():
    text = render_document(_pdf_doc())
    assert "--- Page 1 ---" in text
    assert "Hello world" in text
    assert "no text layer" in text


def test_render_grid_keeps_cell_refs_and_skips_empty():
    sheet = Sheet(
        name="S1",
        rows=(
            (Cell("a", "A1"), Cell(5, "B1")),
            (Cell(None, "A2"), Cell("", "B2")),
        ),
        n_rows=2,
        n_cols=2,
    )
    doc = ParsedDocument(kind="xlsx", filename="g.xlsx", sheets=[sheet])
    text = render_document(doc)
    assert "--- Sheet: S1 ---" in text
    assert "A1=a | B1=5" in text
    assert "A2=" not in text


def test_render_docx_paragraphs_and_tables():
    table = TableBlock(
        source="table 1",
        headers=("H1", "H2"),
        rows=((Cell("a", "r1c1"), Cell("b", "r1c2")),),
    )
    doc = ParsedDocument(
        kind="docx",
        filename="d.docx",
        sections=[Section(index=0, kind="table", table=table), Section(index=1, kind="paragraph", text="Para one")],
    )
    text = render_document(doc)
    assert "[table 1]" in text
    assert "H1 | H2" in text
    assert "a | b" in text
    assert "Para one" in text


def test_render_multiline_cell_values_flattened():
    sheet = Sheet(name="S", rows=((Cell("line1\nline2", "A1"),),))
    text = render_document(ParsedDocument(kind="csv", sheets=[sheet]))
    assert "line1 line2" in text
    assert "\nline2" not in text


def test_primary_schema_prompt_adds_record_completeness_rule():
    schema = normalize_schema(
        {
            "name": "rows",
            "fields": {
                "row_id": {"name": "Row Id", "description": "Identifier.", "primary": True},
                "alpha": {"name": "Alpha", "description": "Value."},
            },
        }
    )
    system = build_messages(schema, _pdf_doc())[0]["content"]
    assert "RECORD COMPLETENESS" in system
    assert "EVERY record" in system
    plain = normalize_schema({"fields": {"alpha": {"name": "Alpha", "description": ""}}})
    system_plain = build_messages(plain, _pdf_doc())[0]["content"]
    assert "RECORD COMPLETENESS" not in system_plain
