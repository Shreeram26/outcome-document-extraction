from __future__ import annotations

import json

import pytest

from app.evaluation import (
    evaluate_corpus,
    evaluate_document,
    field_result_to_dict,
    load_manifest,
    result_to_contract,
    select_items,
)
from app.extraction import ExtractionEngine, LlmGateway, normalize_schema
from app.parsing.models import Page, ParsedDocument
from llm_fakes import fake_response, make_transport

from tools import validate

SAMPLES = pytest.importorskip  # noqa: F841  (placeholder to keep flake quiet)


def _engine_with(content, tokens_in=100, tokens_out=40):
    transport, stub = make_transport([fake_response(content, tokens_in, tokens_out)])
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    return engine, stub


def _engine_no_llm():
    transport, stub = make_transport([])
    return ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash"), stub


def _pdf_doc(text):
    return ParsedDocument(kind="pdf", filename="d.pdf", pages=[Page(number=1, text=text)])


SCHEMA_RAW = {
    "name": "widgets",
    "fields": {
        "widget_name": {"name": "Widget Name", "description": "The name."},
        "widget_total": {"name": "Widget Total", "description": "Total number."},
    },
}
TRUTH = {
    "schema": "widgets",
    "entries": [
        {
            "widget_name": {"value": "Alpha", "citation": "[Page 1] Widget Name: Alpha", "confidence": 1.0},
            "widget_total": {"value": 7, "citation": "[Page 1] Widget Total: 7", "confidence": 1.0},
        }
    ],
    "metadata": {"model_calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0, "latency_ms": 0},
}


def _corpus_item(tmp_path):
    schema_path = tmp_path / "widgets.json"
    schema_path.write_text(json.dumps(SCHEMA_RAW))
    truth_path = tmp_path / "d.json"
    truth_path.write_text(json.dumps(TRUTH))
    from app.evaluation import CorpusItem

    return CorpusItem(
        document_id="d", file=tmp_path / "d.pdf", schema_name="widgets",
        schema_path=schema_path, truth_path=truth_path,
    )


def _write_pdf(tmp_path, text):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    p = tmp_path / "d.pdf"
    doc.save(str(p))
    doc.close()
    return p


def test_deterministic_only_document_needs_no_llm_and_scores_full(tmp_path):
    pdf = _write_pdf(tmp_path, "Widget Name: Alpha\nWidget Total: 7")
    item = _corpus_item(tmp_path)
    item.file = pdf
    engine, stub = _engine_no_llm()
    contract, report = evaluate_document(engine, item)
    assert stub.calls == []
    assert report["contract_valid"] is True
    assert report["mode"] == "single"
    assert report["validator_score"] == 1.0
    assert report["field_accuracy"] == {"widget_name": 1.0, "widget_total": 1.0}
    assert report["field_provenance"]["deterministic"] == ["widget_name", "widget_total"]
    assert report["model_calls"] == 0
    assert contract["schema"] == "widgets"
    assert contract["metadata"]["model_calls"] == 0


def test_llm_fallback_split_is_reported(tmp_path):
    pdf = _write_pdf(tmp_path, "Widget Name: Alpha\nunrelated text")
    item = _corpus_item(tmp_path)
    item.file = pdf
    engine, stub = _engine_with(
        json.dumps({"entries": [{"widget_total": {"value": 7, "citation": "[Page 1] 7", "confidence": 0.9}}]})
    )
    contract, report = evaluate_document(engine, item)
    assert len(stub.calls) == 1
    assert report["field_provenance"]["deterministic"] == ["widget_name"]
    assert report["field_provenance"]["llm"] == ["widget_total"]
    assert report["validator_score"] == 1.0
    assert report["model_calls"] == 1
    assert report["tokens_in"] == 100


def test_unresolved_fields_reported_as_null(tmp_path):
    pdf = _write_pdf(tmp_path, "nothing relevant")
    item = _corpus_item(tmp_path)
    item.file = pdf
    engine, stub = _engine_with(json.dumps({"entries": [{"widget_name": None, "widget_total": None}]}))
    contract, report = evaluate_document(engine, item)
    entry = contract["entries"][0]
    for fr in entry.values():
        assert fr["value"] is None
        assert fr["citation"] == ""
        assert fr["confidence"] == 0.0
    assert report["field_provenance"]["unresolved"] == ["widget_name", "widget_total"]
    assert report["missed_cells"] == 2


def test_harness_uses_the_assignment_validator_for_accuracy(tmp_path):
    pdf = _write_pdf(tmp_path, "Widget Name: Alpha\nWidget Total: 999")
    item = _corpus_item(tmp_path)
    item.file = pdf
    engine, _ = _engine_no_llm()
    contract, report = evaluate_document(engine, item)
    manual = validate.score(contract, TRUTH, SCHEMA_RAW)
    assert report["validator_score"] == manual["score"]
    assert report["field_accuracy"]["widget_total"] == 0.0
    assert report["field_accuracy"]["widget_name"] == 1.0
    assert report["hallucinated_cells"] == 0
    assert report["incorrect_cells"] >= 1
    assert report["missed_cells"] == 0


def test_contract_output_passes_validator_contract_check(tmp_path):
    pdf = _write_pdf(tmp_path, "Widget Name: Alpha\nWidget Total: 7")
    item = _corpus_item(tmp_path)
    item.file = pdf
    engine, _ = _engine_no_llm()
    contract, report = evaluate_document(engine, item)
    assert validate.validate_contract(contract, SCHEMA_RAW) == []
    assert set(contract) <= {"schema", "entries", "metadata", "document"}


def test_field_serializer_contract_shape():
    from app.extraction.responses import FieldResult

    fr = FieldResult(value=1425, citation="[Page 1] 1,425", confidence=0.9, page_numbers=[1])
    d = field_result_to_dict(fr)
    assert d == {"value": 1425, "citation": "[Page 1] 1,425", "confidence": 0.9, "page_numbers": [1]}
    null_fr = FieldResult(value=None, citation="x", confidence=0.5, source_locations=["S!A1"])
    d2 = field_result_to_dict(null_fr)
    assert d2 == {"value": None, "citation": "", "confidence": 0.0}
    assert "page_numbers" not in d2 and "source_locations" not in d2


def test_currency_string_values_are_coerced_not_emitted():
    schema = normalize_schema({"fields": {"total": {"name": "Total", "description": ""}}})
    doc = ParsedDocument(kind="pdf", filename="d.pdf", pages=[Page(1, "Total: $1,425")])
    result = ExtractionEngine(_engine_no_llm()[0], "google/gemini-3.8-flash").extract(schema, doc)
    contract = result_to_contract(result, "widgets", latency_ms=5)
    assert contract["entries"][0]["total"]["value"] == 1425
    assert validate.validate_contract(contract, {"fields": {"total": {"name": "Total"}}}) == []


def test_corpus_report_shape_and_aggregates(tmp_path):
    pdf = _write_pdf(tmp_path, "Widget Name: Alpha\nWidget Total: 7")
    item = _corpus_item(tmp_path)
    item.file = pdf
    engine, _ = _engine_no_llm()
    report = evaluate_corpus(engine, [item])
    assert set(report) == {"documents", "overall"}
    overall = report["overall"]
    for key in (
        "mean_validator_score", "row_accuracy_mean", "model_calls", "tokens_in",
        "tokens_out", "cost_usd", "latency_ms_total", "provenance", "worst_fields",
    ):
        assert key in overall
    assert overall["mean_validator_score"] == 1.0
    assert json.dumps(report)  # machine-readable: serializable


def test_manifest_loads_sample_corpus():
    items = load_manifest()
    assert len(items) == 8
    by_id = {i.document_id: i for i in items}
    assert by_id["boxscore_001"].schema_name == "boxscore"
    assert by_id["boxscore_001"].file.name == "boxscore_001.csv"
    assert by_id["rentroll_001"].truth_path is not None
    assert all(i.file.exists() for i in items)
    assert all(i.truth_path and i.truth_path.exists() for i in items)
    assert all(i.schema_path.exists() for i in items)


def test_skip_llm_engine_degrades_without_raising(tmp_path):
    pdf = _write_pdf(tmp_path, "Widget Name: Alpha")
    item = _corpus_item(tmp_path)
    item.file = pdf
    from app.evaluation import build_engine

    engine = build_engine("google/gemini-3.8-flash", skip_llm=True)
    contract, report = evaluate_document(engine, item)
    assert report["contract_valid"] is True
    assert report["model_calls"] == len(report["call_log"])
    assert all(c["status"] == "error" for c in report["call_log"])
    assert report["field_provenance"]["deterministic"] == ["widget_name"]
    assert any("llm fallback failed" in w for w in report["warnings"])


def test_run_cost_cap_refuses_calls_and_trips(tmp_path):
    from app.evaluation import apply_run_cost_cap

    pdf = _write_pdf(tmp_path, "Widget Name and Widget Total are mentioned here")
    item = _corpus_item(tmp_path)
    item.file = pdf
    transport, stub = make_transport([fake_response('{"entries": []}', cost=0.01)] * 10)
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    cap = apply_run_cost_cap(engine, 0.015)
    contract, report = evaluate_document(engine, item)
    assert cap.tripped is True
    assert cap.spent >= 0.015
    assert cap.refused_calls >= 1
    successful = [c for c in report["call_log"] if c["status"] == "success"]
    refused = [c for c in report["call_log"] if c["status"] == "error"]
    assert len(successful) == 2
    assert len(refused) >= 1
    assert any("run cost cap" in w for w in report["warnings"])
    assert report["contract_valid"] is True
    assert report["field_provenance"]["unresolved"] == ["widget_name", "widget_total"]


def test_run_cost_cap_not_tripped_under_threshold(tmp_path):
    from app.evaluation import apply_run_cost_cap

    pdf = _write_pdf(tmp_path, "Widget Name and Widget Total are mentioned here")
    item = _corpus_item(tmp_path)
    item.file = pdf
    transport, stub = make_transport([fake_response('{"entries": []}', cost=0.001)] * 10)
    engine = ExtractionEngine(LlmGateway(transport=transport), model="google/gemini-3.8-flash")
    cap = apply_run_cost_cap(engine, 0.15)
    engine.max_calls = 3
    evaluate_document(engine, item)
    assert cap.tripped is False
    assert abs(cap.spent - 0.003) < 1e-9


def test_select_items_filters_and_validates():
    items = load_manifest()
    assert len(select_items(items, None)) == len(items)
    one = select_items(items, ["budget_001"])
    assert len(one) == 1 and one[0].document_id == "budget_001"
    two = select_items(items, ["budget_001", "om_001"])
    assert [i.document_id for i in two] == ["budget_001", "om_001"]
    with pytest.raises(ValueError, match="unknown document"):
        select_items(items, ["nope_999"])
