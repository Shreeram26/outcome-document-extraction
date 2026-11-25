"""Tests for scanned-page vision extraction. All extraction is mocked at the
gateway boundary — no API credits are used."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pymupdf
import pytest

from app.extraction import ExtractionEngine, LlmGateway, normalize_schema
from app.extraction.ledger import CostLedger
from app.extraction.responses import FieldResult
from llm_fakes import fake_response, make_transport

SCHEMA = normalize_schema({
    "name": "lease",
    "fields": {
        "tenant_name": {"name": "Tenant Name", "description": "Full legal name."},
        "lease_term_months": {"name": "Lease Term", "description": "Term in months."},
        "base_rent": {"name": "Base Rent", "description": "Monthly base rent."},
    },
})


def _make_scanned_pdf(tmp_path, pages=3):
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10))
        page.insert_image(page.rect, pixmap=pix)
    path = str(tmp_path / "lease.pdf")
    doc.save(path)
    doc.close()
    return path


def _engine(content, **kwargs):
    transport, stub = make_transport([fake_response(content)] * 80)
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="anthropic/claude-haiku-5.5", **kwargs)
    return engine, stub


def test_fully_scanned_pdf_invokes_vision(tmp_path):
    pdf = _make_scanned_pdf(tmp_path, pages=3)
    from app.parsing import parse_file
    doc = parse_file(pdf)
    assert doc.scanned_pages == [1, 2, 3]
    content = json.dumps({"entries": [
        {"tenant_name": "ACME Corp", "lease_term_months": 60, "base_rent": 2500},
    ]})
    engine, stub = _engine(content)
    result = engine.extract(SCHEMA, doc)
    assert len(stub.calls) >= 1
    assert result.entries[0]["tenant_name"].value == "ACME Corp"


def test_digital_pdf_does_not_invoke_vision(tmp_path):
    """A text PDF goes through the normal text pipeline (no vision)."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Tenant shall pay rent monthly.")
    pdf = str(tmp_path / "text.pdf")
    doc.save(pdf)
    doc.close()
    from app.parsing import parse_file
    parsed = parse_file(pdf)
    assert not parsed.scanned_pages
    content = json.dumps({"entries": [{"tenant_name": "Text Corp"}]})
    engine, stub = _engine(content)
    result = engine.extract(SCHEMA, parsed)
    assert len(result.entries) >= 1


def test_vision_multi_record(tmp_path):
    pdf = _make_scanned_pdf(tmp_path, pages=3)
    from app.parsing import parse_file
    doc = parse_file(pdf)
    content = json.dumps({"entries": [
        {"tenant_name": "ACME", "lease_term_months": 60, "base_rent": 2500},
        {"tenant_name": "BETA", "lease_term_months": 36, "base_rent": 1800},
    ]})
    engine, _ = _engine(content)
    result = engine.extract(SCHEMA, doc)
    assert len(result.entries) >= 1
    assert result.entries[0]["tenant_name"].value is not None


def test_vision_nulls_for_missing(tmp_path):
    pdf = _make_scanned_pdf(tmp_path, pages=3)
    from app.parsing import parse_file
    doc = parse_file(pdf)
    content = json.dumps({"entries": [
        {"tenant_name": "ACME", "lease_term_months": None, "base_rent": None},
    ]})
    engine, _ = _engine(content)
    result = engine.extract(SCHEMA, doc)
    assert result.entries[0]["tenant_name"].value == "ACME"
    assert result.entries[0]["lease_term_months"].value is None
    assert result.entries[0]["lease_term_months"].citation == ""
    assert result.entries[0]["lease_term_months"].confidence == 0.0


def test_vision_error_degrades(tmp_path):
    pdf = _make_scanned_pdf(tmp_path, pages=3)
    from app.parsing import parse_file
    doc = parse_file(pdf)

    class FailingTransport:
        def create(self, **kwargs):
            raise RuntimeError("gateway unavailable")

    transport = SimpleNamespace(chat=SimpleNamespace(completions=FailingTransport()))
    engine = ExtractionEngine(LlmGateway(transport=transport), model="anthropic/claude-haiku-5.5")
    result = engine.extract(SCHEMA, doc)
    assert result.entries[0]["tenant_name"].value is None
    assert len(result.warnings) > 0


def test_vision_budget_refusal(tmp_path):
    pdf = _make_scanned_pdf(tmp_path, pages=3)
    from app.parsing import parse_file
    doc = parse_file(pdf)
    ledger = CostLedger(cap_usd=0.001)
    engine = ExtractionEngine(LlmGateway(ledger=ledger), model="anthropic/claude-haiku-5.5")
    result = engine.extract(SCHEMA, doc)
    assert result.entries[0]["tenant_name"].value is None


def test_vision_generic_schema(tmp_path):
    pdf = _make_scanned_pdf(tmp_path, pages=2)
    from app.parsing import parse_file
    doc = parse_file(pdf)
    schema = normalize_schema({
        "name": "invoice",
        "fields": {
            "invoice_number": {"name": "Invoice Number", "description": "ID."},
            "total_amount": {"name": "Total", "description": "Total."},
        },
    })
    content = json.dumps({"entries": [{"invoice_number": "INV-001", "total_amount": 500}]})
    engine, _ = _engine(content)
    result = engine.extract(schema, doc)
    assert result.entries[0]["invoice_number"].value == "INV-001"


def _vision_response(fields=None):
    if fields is None:
        fields = {"tenant_name": "ACME Corp", "lease_term_months": 60, "base_rent": 2500}
    return fake_response(json.dumps({"entries": [fields]}), tokens_in=8000, tokens_out=2000)
