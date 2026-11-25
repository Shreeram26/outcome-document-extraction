from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from app.evaluation import field_result_to_dict, result_to_contract
from app.extraction import (
    ExtractionEngine,
    LlmGateway,
    LlmTimeoutError,
    normalize_schema,
    render_document,
)
from app.parsing.models import Cell, Page, ParsedDocument, Sheet
from llm_fakes import fake_response, make_transport

SCHEMA = normalize_schema(
    {
        "name": "widgets",
        "fields": {
            "alpha": {"name": "Alpha", "description": "The 'Alpha' line."},
            "beta": {"name": "Beta", "description": "The 'Beta' line."},
        },
    }
)


def _pdf_doc(pages):
    return ParsedDocument(kind="pdf", filename="d.pdf", pages=[
        Page(number=i + 1, text=text) for i, text in enumerate(pages)
    ])


def _engine(content, **gateway_kwargs):
    transport, stub = make_transport([fake_response(content)] * 20)
    gateway = LlmGateway(transport=transport, **gateway_kwargs)
    return ExtractionEngine(gateway, model="google/gemini-3.8-flash"), stub


def test_watchdog_stops_hanging_call_and_job_continues():
    class HangingTransport:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            time.sleep(3)
            return fake_response("{}")

    transport = SimpleNamespace(chat=SimpleNamespace(completions=HangingTransport()))
    gateway = LlmGateway(transport=transport, wall_clock=0.3)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    doc = _pdf_doc(["nothing useful here"])
    started = time.monotonic()
    result = engine.extract(SCHEMA, doc)
    elapsed = time.monotonic() - started
    assert elapsed < 2.5
    assert result.entries[0]["alpha"].value is None
    assert result.call_log and result.call_log[0]["status"] == "timeout"
    assert any("timeout" in w for w in result.warnings)


def test_timeout_recorded_in_contract_metadata():
    class HangingTransport:
        def create(self, **kwargs):
            time.sleep(2)
            return fake_response("{}")

    transport = SimpleNamespace(chat=SimpleNamespace(completions=HangingTransport()))
    gateway = LlmGateway(transport=transport, wall_clock=0.2)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(SCHEMA, _pdf_doc(["nothing useful here"]))
    contract = result_to_contract(result, "widgets", latency_ms=10)
    assert contract["metadata"]["model_calls"] >= 1
    assert any(c["status"] == "timeout" for c in contract["metadata"]["calls"])


def test_large_document_uses_targeted_context_not_whole_document():
    rows = [(Cell("Header", "A1"), Cell("Value", "B1"))]
    for i in range(2, 600):
        label = "Alpha" if i % 40 == 0 else f"filler row {i} with padding words"
        rows.append((Cell(label, f"A{i}"), Cell(i, f"B{i}")))
    doc = ParsedDocument(
        kind="xlsx", filename="big.xlsx",
        sheets=[Sheet(name="Big", rows=tuple(rows), n_rows=len(rows), n_cols=2)],
    )
    assert len(render_document(doc)) > 25000
    content = json.dumps({"entries": [{"alpha": 5}]})
    engine, stub = _engine(content)
    result = engine.extract(SCHEMA, doc)
    assert 1 <= len(stub.calls) <= 2
    assert all(c["context_chars"] < 8000 for c in result.call_log)
    assert result.call_log[0]["fields"] == ["alpha", "beta"]
    assert all(c["fields"] == ["beta"] for c in result.call_log[1:])
    assert result.entries[0]["alpha"].value == 5
    assert result.provenance["alpha"] == "llm"


def test_progressive_expansion_on_second_stage():
    pages = [
        "Alpha: 1 " + "filler " * 700,
        "Beta: 2 " + "filler " * 200,
    ]
    doc = _pdf_doc(pages)
    transport, stub = make_transport([
        fake_response(json.dumps({"entries": [{"alpha": 1}]})),
        fake_response(json.dumps({"entries": [{"beta": 2}]})),
    ] * 5)
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    result = engine.extract(SCHEMA, doc)
    assert len(stub.calls) == 2
    sizes = [c["context_chars"] for c in result.call_log]
    assert sizes[1] > sizes[0]
    assert result.call_log[0]["status"] == "success"
    assert result.call_log[1]["status"] == "success"
    assert result.entries[0]["alpha"].value == 1
    assert result.entries[0]["beta"].value == 2
    assert result.provenance["alpha"] == "llm" and result.provenance["beta"] == "llm"


def test_call_limit_stops_fallback_gracefully():
    doc = _pdf_doc(["nothing useful " * 100])
    engine, stub = _engine("{}")
    engine.max_calls = 2
    result = engine.extract(SCHEMA, doc)
    assert len(stub.calls) == 2
    assert len(result.call_log) == 2
    assert any("call budget reached" in w for w in result.warnings)
    assert result.entries[0]["alpha"].value is None
    assert result.provenance["alpha"] == "unresolved"


def test_cost_limit_stops_fallback():
    doc = _pdf_doc(["nothing useful " * 100 + " Alpha and Beta mentioned"])
    transport, stub = make_transport([fake_response("{}")] * 20)
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    engine.max_cost_usd = 0.11
    result = engine.extract(SCHEMA, doc)
    assert len(stub.calls) == 2
    assert result.call_log[-1]["cumulative_cost_usd"] > 0
    assert any("cost budget reached" in w for w in result.warnings)
    assert result.entries[0]["alpha"].value is None


def test_chunks_receive_only_relevant_fields_and_skip_irrelevant_chunks():
    pages = [
        "Page heading with no schema words " + "padding " * 400,
        "Alpha: 7 " + "padding " * 300,
        "Beta: 9 " + "padding " * 300,
    ]
    doc = _pdf_doc(pages)
    engine, stub = _engine("{}")
    engine.chunk_chars = 3000
    result = engine.extract(SCHEMA, doc)
    field_sets = [tuple(sorted(c["fields"])) for c in result.call_log]
    assert ("alpha",) in field_sets or ("alpha", "beta") in field_sets
    assert ("beta",) in field_sets
    assert all(c["context_chars"] <= 3200 for c in result.call_log)
    assert result.entries[0]["alpha"].value is None


def test_grid_duplicate_primary_rows_conflict_detected():
    schema = normalize_schema(
        {
            "fields": {
                "code": {"name": "Code", "description": "The 'Code' column.", "primary": True},
                "qty": {"name": "Qty", "description": "Quantity."},
            }
        }
    )
    sheet = Sheet(
        name="S",
        rows=(
            (Cell("misc", "A1"), Cell("data", "B1")),
            (Cell("misc", "A2"), Cell("data", "B2")),
            (Cell("misc", "A3"), Cell("data", "B3")),
        ),
    )
    doc = ParsedDocument(kind="xlsx", filename="s.xlsx", sheets=[sheet])
    content = json.dumps(
        {
            "entries": [
                {"code": "A-1", "qty": 1},
                {"code": "A-1", "qty": 2},
                {"code": "A-2", "qty": 5},
            ]
        }
    )
    engine, _ = _engine(content)
    result = engine.extract(schema, doc)
    assert len(result.entries) == 2
    assert result.entries[0]["code"].value == "A-1"
    assert result.entries[0]["qty"].value == 1
    assert any("A-1" in c and "duplicate" in c for c in result.conflicts)


def test_scanned_document_never_calls_llm():
    doc = ParsedDocument(kind="pdf", filename="scan.pdf", pages=[Page(1, ""), Page(2, "  ")])
    engine, stub = _engine("{}")
    result = engine.extract(SCHEMA, doc)
    assert stub.calls == []
    assert any("no extractable text" in w for w in result.warnings)
    assert result.entries[0]["alpha"].value is None
    assert result.provenance["alpha"] == "unresolved"


def test_repeated_gateway_failures_trip_circuit_breaker():
    class FailingTransport:
        def create(self, **kwargs):
            raise RuntimeError("connection refused")

    transport = SimpleNamespace(chat=SimpleNamespace(completions=FailingTransport()))
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    result = engine.extract(SCHEMA, _pdf_doc(["Alpha: 1"]))
    assert len(result.call_log) == 2
    assert all(c["status"] == "error" for c in result.call_log)
    assert any("consecutively" in w for w in result.warnings)


def test_call_log_observability_fields():
    doc = _pdf_doc(["Alpha: 1"])
    engine, _ = _engine(json.dumps({"entries": [{"alpha": 1}]}), )
    result = engine.extract(SCHEMA, doc)
    record = result.call_log[0]
    for key in ("call", "model", "fields", "context_chars", "duration_ms", "status", "tokens_in", "tokens_out", "reserved_usd", "cost_usd", "cumulative_cost_usd"):
        assert key in record
    assert record["status"] == "success"
    assert record["call"] == 1
    assert record["reserved_usd"] > 0
    assert record["cost_usd"] == 0.0
    assert record["cumulative_cost_usd"] > 0


def test_metadata_includes_call_log_and_conflicts():
    doc = _pdf_doc(["Alpha: 1"])
    engine, _ = _engine(json.dumps({"entries": [{"alpha": 1}]}))
    result = engine.extract(SCHEMA, doc)
    contract = result_to_contract(result, "widgets", latency_ms=5)
    assert contract["metadata"]["model_calls"] == len(result.call_log)
    assert contract["metadata"]["calls"] == result.call_log


def test_grid_fallback_uses_row_bounded_chunks():
    """Regression: the grid fallback previously chunked at chunk_chars (12k),
    producing ~60-row batches that truncated at the completion budget. Grid
    chunks must respect full_chunk_chars so every batch parses."""
    rows = [(Cell("misc", "A1"), Cell("data", "B1"))]
    for i in range(2, 122):
        rows.append((Cell(f"K-{i}", f"A{i}"), Cell(i, f"B{i}")))
    doc = ParsedDocument(
        kind="xlsx", filename="g.xlsx",
        sheets=[Sheet(name="S", rows=tuple(rows), n_rows=len(rows), n_cols=2)],
    )
    schema = normalize_schema(
        {
            "name": "keys",
            "fields": {
                "code": {"name": "Code", "description": "Identifier.", "primary": True},
                "amount": {"name": "Amount", "description": "Amount."},
            },
        }
    )
    all_rows = json.dumps({"entries": [
        {"code": f"K-{n}", "amount": n} for n in range(2, 122)
    ]})
    transport, stub = make_transport([fake_response(all_rows)] * 60)
    engine = ExtractionEngine(
        LlmGateway(transport=transport), model="google/gemini-3.8-flash",
        full_chunk_chars=200, max_calls=40,
    )
    result = engine.extract(schema, doc)
    assert len(stub.calls) >= 2
    assert all(len(c["messages"][1]["content"]) <= 260 for c in stub.calls)
    assert len(result.entries) == 120
    assert result.entries[0]["code"].value == "K-2"
    assert result.entries[-1]["code"].value == "K-121"


DEFAULTS_SCHEMA = normalize_schema(
    {
        "name": "lease-ish",
        "fields": {
            "item_id": {"name": "Item Id", "description": "The 'Item Id' column.", "primary": True},
            "flag": {"name": "Flag", "description": "true if required, false otherwise."},
            "count": {"name": "Count", "description": "How many, as a number. 0 if none."},
            "tags": {"name": "Tags", "description": "List of tags. Empty list if none."},
            "note": {"name": "Note", "description": "Free note. null if not stated."},
        },
    }
)


def test_defaults_applied_when_llm_returns_null():
    doc = _pdf_doc(["nothing useful"])
    content = json.dumps({"entries": [{"item_id": "T-1", "flag": None, "count": None, "tags": None, "note": None}]})
    transport, stub = make_transport([fake_response(content)] * 5)
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(DEFAULTS_SCHEMA, doc)
    entry = result.entries[0]
    assert entry["flag"].value is False
    assert entry["count"].value == 0
    assert entry["tags"].value == []
    assert entry["note"].value is None
    assert entry["flag"].citation == ""
    assert result.provenance["flag"] == "default"
    assert any("schema defaults applied" in w for w in result.warnings)


def test_defaults_do_not_override_extracted_values():
    doc = _pdf_doc(["nothing useful"])
    content = json.dumps({"entries": [
        {"item_id": "T-1", "flag": True, "count": 5, "tags": ["a"], "note": "hello"},
    ]})
    transport, _ = make_transport([fake_response(content)] * 5)
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(DEFAULTS_SCHEMA, doc)
    entry = result.entries[0]
    assert entry["flag"].value is True
    assert entry["count"].value == 5
    assert entry["tags"].value == ["a"]
    assert entry["note"].value == "hello"
    assert "default" not in result.provenance.values()


def test_defaults_skipped_for_type_incompatible_field():
    doc = _pdf_doc(["nothing useful"])
    content = json.dumps({"entries": [
        {"item_id": "T-1", "count": "five"},
        {"item_id": "T-2"},
    ]})
    transport, _ = make_transport([fake_response(content)] * 5)
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(DEFAULTS_SCHEMA, doc)
    by_id = {e["item_id"].value: e for e in result.entries}
    assert by_id["T-1"]["count"].value == "five"
    assert by_id["T-2"]["count"].value is None
    assert "count" not in [k for k, v in result.provenance.items() if v == "default"]


def test_defaults_skipped_when_gateway_broken():
    class FailingTransport:
        def create(self, **kwargs):
            raise RuntimeError("connection refused")

    transport = SimpleNamespace(chat=SimpleNamespace(completions=FailingTransport()))
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(DEFAULTS_SCHEMA, _pdf_doc(["nothing useful"]))
    assert result.entries[0]["flag"].value is None
    assert all(v != "default" for v in result.provenance.values())


def test_defaults_skipped_for_textless_document():
    doc = ParsedDocument(kind="pdf", filename="scan.pdf", pages=[Page(1, "")])
    transport, stub = make_transport([fake_response('{"entries": []}')] * 3)
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(DEFAULTS_SCHEMA, doc)
    assert stub.calls == []
    assert result.entries[0]["flag"].value is None
    assert all(v != "default" for v in result.provenance.values())
