from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.extraction import ExtractionEngine, LlmBudgetError, LlmGateway, normalize_schema
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


def _response(cost):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"entries": []}'))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=200, cost=cost),
    )


class ScriptedTransport:
    """Inner transport whose create() sleeps before returning, to simulate
    requests that finish after the watchdog abandoned them."""

    def __init__(self, cost, delay):
        self.cost = cost
        self.delay = delay
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        time.sleep(self.delay)
        return _response(self.cost)


def test_successful_call_reserves_then_settles():
    ledger = CostLedger(cap_usd=1.0)
    reservation = ledger.try_reserve(0.05, model="m")
    assert reservation is not None
    assert ledger.exposure_usd == pytest.approx(0.05)
    assert ledger.outstanding_usd == pytest.approx(0.05)
    assert ledger.settled_usd == 0.0
    assert ledger.complete(reservation, 0.03) == "settled"
    assert ledger.settled_usd == pytest.approx(0.03)
    assert ledger.exposure_usd == pytest.approx(0.03)


def test_timeout_retains_reservation_and_late_completion_reconciles():
    ledger = CostLedger(cap_usd=1.0)
    reservation = ledger.try_reserve(0.05, model="m")
    assert ledger.abandon(reservation) == "abandoned"
    assert ledger.exposure_usd == pytest.approx(0.05)
    assert ledger.settled_usd == 0.0
    assert ledger.reconcile(reservation, 0.127549) == "reconciled"
    assert ledger.settled_usd == pytest.approx(0.127549)
    assert ledger.outstanding_usd == 0.0
    assert ledger.exposure_usd == pytest.approx(0.127549)
    assert ledger.reconcile(reservation, 0.5) == "ignored"


def test_reservation_blocks_subsequent_call():
    ledger = CostLedger(cap_usd=0.06)
    assert ledger.try_reserve(0.05, model="m") is not None
    assert ledger.try_reserve(0.05, model="m") is None
    assert ledger.refused_calls == 1
    assert ledger.tripped is True


def test_concurrent_reservations_do_not_lose_updates():
    ledger = CostLedger(cap_usd=100.0)
    barrier = threading.Barrier(8)

    def worker(n):
        barrier.wait()
        booked = 0.0
        for _ in range(25):
            r = ledger.try_reserve(0.01, model="m")
            assert r is not None
            ledger.complete(r, 0.005)
            booked += 0.005
        return booked

    with ThreadPoolExecutor(max_workers=8) as pool:
        totals = list(pool.map(worker, range(8)))
    assert ledger.settled_usd == pytest.approx(sum(totals))
    assert ledger.settled_usd == pytest.approx(8 * 25 * 0.005)
    assert ledger.outstanding_usd == 0.0
    assert ledger.exposure_usd == pytest.approx(1.0)
    assert ledger.refused_calls == 0


def test_actual_cost_exceeding_reservation_is_booked_immediately():
    ledger = CostLedger(cap_usd=0.2)
    reservation = ledger.try_reserve(0.05, model="m")
    assert ledger.complete(reservation, 0.09) == "settled"
    assert ledger.settled_usd == pytest.approx(0.09)
    assert ledger.exposure_usd == pytest.approx(0.09)
    assert ledger.try_reserve(0.15, model="m") is None


def test_budget_exhaustion_refuses_and_reports():
    ledger = CostLedger(cap_usd=0.05)
    r1 = ledger.try_reserve(0.03, model="m")
    assert ledger.complete(r1, 0.03) == "settled"
    assert ledger.try_reserve(0.03, model="m") is None
    status = ledger.status()
    assert status["refused_calls"] == 1
    assert status["tripped"] is True
    assert status["exposure_usd"] == pytest.approx(0.03)


def test_abandoned_failure_retains_reservation():
    ledger = CostLedger(cap_usd=1.0)
    r = ledger.try_reserve(0.05, model="m")
    ledger.abandon(r)
    assert ledger.failure(r) == "retained"
    assert ledger.exposure_usd == pytest.approx(0.05)


def test_active_failure_releases_reservation():
    ledger = CostLedger(cap_usd=1.0)
    r = ledger.try_reserve(0.05, model="m")
    assert ledger.failure(r) == "released"
    assert ledger.exposure_usd == 0.0


def test_unknown_provider_cost_retains_reservation():
    ledger = CostLedger(cap_usd=1.0)
    r = ledger.try_reserve(0.05, model="m")
    assert ledger.complete(r, None) == "settled_unknown"
    assert ledger.exposure_usd == pytest.approx(0.05)
    assert ledger.settled_usd == 0.0


def test_no_cap_never_refuses():
    ledger = CostLedger(cap_usd=None)
    for _ in range(50):
        assert ledger.try_reserve(10.0, model="m") is not None
    assert ledger.exposure_usd == pytest.approx(500.0)


def test_gateway_reserves_before_call_and_refuses_when_cap_would_be_exceeded():
    transport, stub = make_transport([fake_response('{"entries": []}')] * 10)
    ledger = CostLedger(cap_usd=0.02)
    gateway = LlmGateway(transport=transport, ledger=ledger, default_max_tokens=4000)
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(SCHEMA, _pdf_doc("Alpha and Beta mentioned"))
    assert len(stub.calls) == 1
    assert ledger.refused_calls >= 1
    assert any(c["status"] == "budget_refused" for c in result.call_log)
    assert any(c["status"] == "success" for c in result.call_log)
    assert result.entries[0]["alpha"].value is None


def test_gateway_enforces_max_tokens_and_estimates_grow_with_it():
    transport, stub = make_transport([fake_response('{"entries": []}')] * 4)
    gateway = LlmGateway(transport=transport, default_max_tokens=4000)
    messages = [{"role": "user", "content": "x" * 4000}]
    small = gateway.estimate_cost_usd(messages, "google/gemini-3.8-flash")
    big = gateway.estimate_cost_usd(messages, "google/gemini-3.8-flash", max_tokens=16000)
    assert big > small
    gateway.complete("google/gemini-3.8-flash", messages)
    assert stub.calls[0]["max_tokens"] == 4000


def test_engine_timeout_retains_reservation_in_exposure():
    inner = ScriptedTransport(cost=0.127549, delay=1.0)
    transport = SimpleNamespace(chat=SimpleNamespace(completions=inner))
    gateway = LlmGateway(transport=transport, wall_clock=0.3, ledger=CostLedger(cap_usd=1.0))
    engine = ExtractionEngine(gateway, model="google/gemini-3.8-flash")
    result = engine.extract(SCHEMA, _pdf_doc("nothing useful"))
    assert result.call_log[0]["status"] == "timeout"
    exposure_right_after = result.cost_exposure_usd
    assert exposure_right_after > 0
    assert result.entries[0]["alpha"].value is None
    time.sleep(1.2)
    timeouts = [c for c in result.call_log if c["status"] == "timeout"]
    assert len(timeouts) == 2
    assert gateway.ledger.settled_usd == pytest.approx(2 * 0.127549)
    assert gateway.ledger.outstanding_usd == 0.0
    assert gateway.ledger.exposure_usd == pytest.approx(2 * 0.127549)
    assert result.cost_exposure_usd == pytest.approx(exposure_right_after)


def test_ledger_reconciles_pilot_discrepancy_scenario():
    ledger = CostLedger(cap_usd=0.15)
    reserved = []
    for cost_actual, delay in ((0.070095, 0), (0.063008, 0), (None, 0.4), (0.020396, 0)):
        r = ledger.try_reserve(0.03, model="google/gemini-3.8-flash")
        assert r is not None
        if delay:
            ledger.abandon(r)
        reserved.append((r, cost_actual, delay))
    time.sleep(0.05)
    for r, actual, delay in reserved:
        if delay:
            ledger.reconcile(r, 0.127549)
        else:
            ledger.complete(r, actual)
    assert ledger.exposure_usd == pytest.approx(0.281048)
    assert ledger.settled_usd == pytest.approx(0.281048)
    assert ledger.status()["refused_calls"] == 0
