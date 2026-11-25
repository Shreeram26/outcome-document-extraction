from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from app.extraction import (
    ExtractionEngine,
    LlmBadResponseError,
    LlmBudgetError,
    LlmGateway,
    normalize_schema,
)
from app.extraction.ledger import CostLedger
from app.parsing.models import Page, ParsedDocument
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


def _pdf_doc(text):
    return ParsedDocument(kind="pdf", filename="d.pdf", pages=[Page(number=1, text=text)])


# ---------- Fix 1: provider cost read from response.raw ----------


def test_provider_cost_recorded_in_settled_exposure_and_call_log():
    transport, stub = make_transport([fake_response('{"entries": []}', cost=0.03)] * 5)
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(SCHEMA, _pdf_doc("nothing useful"))
    assert len(stub.calls) == 2
    assert result.cost_settled_usd == pytest.approx(0.06)
    assert all(c["cost_usd"] == pytest.approx(0.03) for c in result.call_log)
    assert result.cost_exposure_usd == pytest.approx(0.06)


def test_provider_cost_flows_into_contract_metadata():
    from app.evaluation import result_to_contract

    transport, _ = make_transport([fake_response('{"entries": []}', cost=0.03)] * 5)
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(SCHEMA, _pdf_doc("nothing useful"))
    contract = result_to_contract(result, "widgets", latency_ms=5)
    assert contract["metadata"]["cost_settled_usd"] == pytest.approx(0.06)
    assert contract["metadata"]["cost_usd"] == pytest.approx(0.06)
    models = contract["metadata"]["models"]["google/gemini-3.8-flash"]
    assert models["settled_cost_usd"] == pytest.approx(0.06)


def test_cost_settled_is_counted_exactly_once_per_call():
    transport, _ = make_transport([fake_response('{"entries": []}', cost=0.03)] * 5)
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(SCHEMA, _pdf_doc("nothing useful"))
    calls = len(result.call_log)
    assert result.cost_settled_usd == pytest.approx(0.03 * calls)


# ---------- Fix 2: finite output limit enforced ----------


def test_default_max_tokens_none_is_rejected():
    with pytest.raises(ValueError, match="default_max_tokens"):
        LlmGateway(transport=SimpleNamespace(), default_max_tokens=None)


def test_non_positive_default_max_tokens_is_rejected():
    with pytest.raises(ValueError, match="default_max_tokens"):
        LlmGateway(transport=SimpleNamespace(), default_max_tokens=0)


def test_non_positive_max_tokens_argument_is_rejected():
    gateway = LlmGateway(transport=SimpleNamespace())
    with pytest.raises(ValueError, match="max_tokens"):
        gateway.complete("google/gemini-3.8-flash", [{"role": "user", "content": "x"}], max_tokens=0)


def test_reservations_stay_meaningful_with_finite_limit():
    ledger = CostLedger(cap_usd=1.0)
    gateway = LlmGateway(transport=SimpleNamespace(), ledger=ledger, default_max_tokens=8000)
    estimate = gateway.estimate_cost_usd(
        [{"role": "user", "content": "x" * 4000}], "google/gemini-3.8-flash"
    )
    assert estimate > 0.005
    reservation = ledger.try_reserve(estimate, model="google/gemini-3.8-flash")
    assert reservation is not None


# ---------- Fix 3: billed call with missing content does not refund ----------


def test_billed_missing_content_does_not_refund_document_budget():
    transport, stub = make_transport([fake_response(None, cost=0.02)] * 5)
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(SCHEMA, _pdf_doc("nothing useful"))
    assert result.call_log[0]["status"] == "error"
    assert result.call_log[0]["cost_usd"] == pytest.approx(0.02)
    assert result.cost_settled_usd == pytest.approx(2 * 0.02)
    assert result.cost_exposure_usd > 0
    assert any("no content" in w for w in result.warnings)


def test_billed_missing_content_settles_run_ledger():
    transport, stub = make_transport([fake_response(None, cost=0.02)] * 5)
    ledger = CostLedger(cap_usd=1.0)
    gateway = LlmGateway(transport=transport, ledger=ledger)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    engine.extract(SCHEMA, _pdf_doc("nothing useful"))
    assert ledger.settled_usd == pytest.approx(2 * 0.02)
    assert ledger.outstanding_usd == 0.0


def test_unbilled_failure_still_refunds_document_budget():
    class FailingTransport:
        def create(self, **kwargs):
            raise RuntimeError("connection refused")

    transport = SimpleNamespace(chat=SimpleNamespace(completions=FailingTransport()))
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(SCHEMA, _pdf_doc("nothing useful"))
    assert result.call_log[0]["cost_usd"] == 0.0
    assert result.cost_settled_usd == 0.0
    assert result.cost_exposure_usd == pytest.approx(0.0)


def test_llm_bad_response_error_carries_cost():
    exc = LlmBadResponseError("no content", actual_cost_usd=0.02)
    assert exc.response_arrived is True
    assert exc.actual_cost_usd == 0.02


# ---------- Fix 4: truncation detection, configurable limit ----------


def _truncated_response():
    response = fake_response('{"entries": [{"alpha": ')
    response.choices[0].finish_reason = "length"
    return response


def test_truncated_response_reported_as_parse_error_with_guidance():
    transport, stub = make_transport([_truncated_response()] * 5)
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(SCHEMA, _pdf_doc("nothing useful"))
    assert result.call_log[0]["status"] == "parse_error"
    assert any("max_tokens" in w for w in result.warnings)


def test_default_max_tokens_is_16000():
    transport, stub = make_transport([fake_response('{"entries": []}')] * 5)
    gateway = LlmGateway(transport=transport)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    engine.extract(SCHEMA, _pdf_doc("nothing useful"))
    assert stub.calls[0]["max_tokens"] == 16000


def test_build_engine_applies_max_tokens():
    from app.evaluation import build_engine

    engine = build_engine("google/gemini-3.8-flash", dry_run=True, max_tokens=12345)
    assert engine.gateway.default_max_tokens == 12345


# ---------- Fix 5: timeout boundary ----------


def test_response_completed_before_processing_is_not_lost():
    """A response that completes within the wall clock is returned normally,
    even when an event-gated transport races the watchdog."""
    release = threading.Event()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"entries": []}'), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=200, cost=0.02),
    )

    class JustInTimeTransport:
        def create(self, **kwargs):
            release.wait(timeout=2.0)
            return response

    transport = SimpleNamespace(chat=SimpleNamespace(completions=JustInTimeTransport()))
    gateway = LlmGateway(transport=transport, wall_clock=0.05)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")

    timer = threading.Timer(0.02, release.set)
    timer.start()
    result = engine.extract(SCHEMA, _pdf_doc("nothing useful"))
    timer.join()
    assert result.call_log[0]["status"] == "success"
    assert result.cost_settled_usd == pytest.approx(0.02 * len(result.call_log))
    assert not any("timeout" in w for w in result.warnings)


def test_late_completion_after_timeout_reconciles_cost():
    inner = SimpleNamespace()

    class LateTransport:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            time.sleep(0.4)
            return fake_response('{"entries": []}', cost=0.127549)

    inner.completions = LateTransport()
    transport = SimpleNamespace(chat=SimpleNamespace(completions=inner.completions))
    gateway = LlmGateway(transport=transport, wall_clock=0.1, ledger=CostLedger(cap_usd=1.0))
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(SCHEMA, _pdf_doc("nothing useful"))
    assert result.call_log[0]["status"] == "timeout"
    assert result.cost_exposure_usd > 0
    time.sleep(0.6)
    assert gateway.ledger.settled_usd == pytest.approx(2 * 0.127549)
    assert gateway.ledger.outstanding_usd == 0.0


def test_extra_params_passthrough_to_transport():
    transport, stub = make_transport([fake_response('{"entries": []}')] * 2)
    gateway = LlmGateway(transport=transport)
    gateway.complete(
        "google/gemini-3.8-flash",
        [{"role": "user", "content": "x"}],
        extra_params={"response_format": {"type": "json_object"}},
    )
    assert stub.calls[0]["response_format"] == {"type": "json_object"}
    gateway.complete("google/gemini-3.8-flash", [{"role": "user", "content": "x"}])
    assert "response_format" not in stub.calls[1]
