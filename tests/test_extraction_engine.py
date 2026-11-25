from __future__ import annotations

import json

from app.extraction import ExtractionEngine, LlmGateway, normalize_schema, render_document
from app.parsing.models import Cell, Page, ParsedDocument, Sheet
from llm_fakes import fake_response, make_transport


def _pdf_doc(lines):
    return ParsedDocument(
        kind="pdf", filename="f.pdf", pages=[Page(number=1, text="\n".join(lines))]
    )


def _sheet_doc(rows):
    return ParsedDocument(
        kind="xlsx", filename="s.xlsx", sheets=[Sheet(name="S1", rows=rows, n_rows=len(rows), n_cols=2)]
    )


def test_no_llm_call_when_everything_resolves_deterministically():
    schema = normalize_schema(
        {"fields": {"alpha": {"name": "Alpha", "description": ""}, "beta": {"name": "Beta", "description": ""}}}
    )
    doc = _pdf_doc(["Alpha: 7", "Beta: hello"])
    transport, stub = make_transport([])
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    result = engine.extract(schema, doc)
    assert stub.calls == []
    assert result.llm is None
    assert result.entries[0]["alpha"].value == 7
    assert result.entries[0]["beta"].value == "hello"
    assert result.entries[0]["alpha"].citation == "[Page 1] Alpha: 7"


def test_no_llm_call_when_table_resolves_fully():
    schema = normalize_schema(
        {
            "fields": {
                "code": {"name": "Code", "description": "The 'Code' column.", "primary": True},
                "qty": {"name": "Qty", "description": "The 'Qty' column."},
            }
        }
    )
    doc = _sheet_doc(
        [
            (Cell("Code", "A1"), Cell("Qty", "B1")),
            (Cell("A-1", "A2"), Cell(3, "B2")),
            (Cell("A-2", "A3"), Cell(4, "B3")),
        ]
    )
    transport, stub = make_transport([])
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    result = engine.extract(schema, doc)
    assert stub.calls == []
    assert result.mode == "table"
    assert len(result.entries) == 2
    assert result.entries[0]["code"].value == "A-1"
    assert result.entries[1]["qty"].value == 4


def test_partial_deterministic_then_llm_fallback_for_unresolved_only():
    schema = normalize_schema(
        {"fields": {"alpha": {"name": "Alpha", "description": ""}, "beta": {"name": "Beta", "description": ""}}}
    )
    doc = _pdf_doc(["Alpha: 7", "unrelated body text"])
    content = json.dumps(
        {"entries": [{"beta": {"value": "found", "citation": "[Page 1] unrelated body text", "confidence": 0.9}}]}
    )
    transport, stub = make_transport([fake_response(content, tokens_in=10, tokens_out=5)])
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    result = engine.extract(schema, doc)
    assert len(stub.calls) == 1
    user_content = stub.calls[0]["messages"][1]["content"]
    assert '"beta"' in user_content
    assert '"alpha"' not in user_content
    assert result.entries[0]["alpha"].value == 7
    assert result.entries[0]["beta"].value == "found"
    assert result.llm.tokens_in == 10
    assert result.mode == "single"


def test_deterministic_values_never_overwritten_by_llm():
    schema = normalize_schema(
        {"fields": {"alpha": {"name": "Alpha", "description": ""}, "beta": {"name": "Beta", "description": ""}}}
    )
    doc = _pdf_doc(["Alpha: 7"])
    content = json.dumps(
        {
            "entries": [
                {
                    "alpha": {"value": 999, "citation": "hallucinated", "confidence": 1.0},
                    "beta": {"value": 2, "citation": "[Page 1] x", "confidence": 0.8},
                }
            ]
        }
    )
    transport, _ = make_transport([fake_response(content)])
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    result = engine.extract(schema, doc)
    assert result.entries[0]["alpha"].value == 7
    assert result.entries[0]["alpha"].citation == "[Page 1] Alpha: 7"
    assert result.entries[0]["beta"].value == 2
    assert any("unknown field 'alpha'" in w for w in result.warnings)


def test_missing_fields_return_null_with_empty_citation():
    schema = normalize_schema(
        {"fields": {"alpha": {"name": "Alpha", "description": ""}, "beta": {"name": "Beta", "description": ""}}}
    )
    doc = _pdf_doc(["nothing useful here"])
    content = json.dumps(
        {"entries": [{"alpha": None, "beta": {"value": None, "citation": "x", "confidence": 0.5}}]}
    )
    transport, _ = make_transport([fake_response(content)])
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    result = engine.extract(schema, doc)
    for key in ("alpha", "beta"):
        fr = result.entries[0][key]
        assert fr.value is None
        assert fr.citation == ""
        assert fr.confidence == 0.0


def test_citations_and_source_locations_preserved_through_merge():
    schema = normalize_schema(
        {"fields": {"rsf": {"name": "Property RSF", "description": "Square footage."}}}
    )
    doc = _sheet_doc([(Cell("Property RSF", "A5"), Cell(339050, "B5"))])
    transport, stub = make_transport([])
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    result = engine.extract(schema, doc)
    fr = result.entries[0]["rsf"]
    assert fr.value == 339050
    assert fr.source_locations == ["S1!B5"]
    assert fr.citation == "[Sheet: S1, Cell: B5] 339050"
    assert stub.calls == []


def test_large_spreadsheet_chunked_never_exceeds_call_cap():
    rows = [(Cell("Header", "A1"), Cell("Value", "B1"))]
    for i in range(2, 600):
        label = "Omega" if i % 40 == 0 else f"filler row {i} with padding words"
        rows.append((Cell(label, f"A{i}"), Cell(i, f"B{i}")))
    doc = _sheet_doc(rows)
    assert len(render_document(doc)) > 25000
    schema = normalize_schema(
        {
            "fields": {
                "omega_count": {"name": "Omega Count", "description": "From the 'Omega' line."},
                "other_thing": {"name": "Other Thing", "description": "Anything about other things."},
            }
        }
    )
    content = json.dumps({"entries": [{"omega_count": 5}]})
    transport, stub = make_transport([fake_response(content)] * 10)
    engine = ExtractionEngine(
        LlmGateway(transport=transport), model="google/gemini-3.8-flash", chunk_chars=10000
    )
    result = engine.extract(schema, doc)
    assert len(stub.calls) > 1
    assert all(
        len(call["messages"][1]["content"]) <= 11000 for call in stub.calls
    )
    assert result.llm_responses  # chunk calls aggregated
    assert result.entries[0]["omega_count"].value == 5
    assert result.entries[0]["other_thing"].value is None


def test_generic_schema_end_to_end_without_domain_terms():
    schema = normalize_schema(
        {
            "name": "widgets",
            "fields": {
                "serial": {"name": "Serial", "description": "The 'Serial' column.", "primary": True},
                "made_on": {"name": "Made On", "description": "Date."},
            },
        }
    )
    doc = _sheet_doc(
        [
            (Cell("Serial", "A1"), Cell("Made On", "B1")),
            (Cell("S-1", "A2"), Cell("2026-01-15", "B2")),
        ]
    )
    transport, stub = make_transport([])
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    result = engine.extract(schema, doc)
    assert stub.calls == []
    assert result.entries[0]["serial"].value == "S-1"
    assert result.entries[0]["made_on"].value == "2026-01-15"


def test_unparseable_llm_response_degrades_to_warnings_not_crash():
    schema = normalize_schema({"fields": {"alpha": {"name": "Alpha", "description": ""}}})
    doc = _pdf_doc(["no labels here"])
    transport, _ = make_transport([fake_response("I cannot help with that, sorry!")])
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    result = engine.extract(schema, doc)
    assert result.entries[0]["alpha"].value is None
    assert any("llm fallback failed" in w and "sorry" in w for w in result.warnings)


def test_every_final_entry_carries_every_schema_field():
    schema = normalize_schema(
        {"fields": {"x": {"name": "X", "description": ""}, "y": {"name": "Y", "description": ""}, "z": {"name": "Z", "description": ""}}}
    )
    doc = _pdf_doc(["X: 1"])
    content = json.dumps({"entries": [{"y": 2}]})
    transport, _ = make_transport([fake_response(content)])
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    result = engine.extract(schema, doc)
    assert set(result.entries[0]) == {"x", "y", "z"}
    assert result.entries[0]["x"].value == 1
    assert result.entries[0]["y"].value == 2
    assert result.entries[0]["z"].value is None
