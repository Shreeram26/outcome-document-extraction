"""Thread-safe cost ledger for LLM budget enforcement.

One ledger instance covers a whole evaluation run. Money is tracked as:

    exposure = settled + outstanding

- `settled`: provider-reported cost of completed responses
- `outstanding`: reservations — pre-call estimates that are either active
  (request in flight), retained (request abandoned after a local timeout;
  the provider may still bill it), or retained-unknown (completed but the
  provider did not report a cost)

A reservation is taken BEFORE the request is launched, so the cap can refuse a
call that would knowingly exceed the budget. Reservations are only released
when a request definitively failed before producing a response; they are
retained on local timeouts and reconciled against the actual bill when an
abandoned request eventually completes."""
from __future__ import annotations

import threading
from typing import Any


class Reservation:
    """Handle to one reserved amount. State transitions happen only inside
    CostLedger methods, under the ledger lock."""

    def __init__(self, ledger: "CostLedger", amount: float, model: str, meta: dict | None):
        self.amount = float(amount)
        self.model = model
        self.meta = dict(meta or {})
        self.state = "reserved"
        self.actual: float | None = None
        self._ledger = ledger

    def __repr__(self) -> str:
        return f"Reservation({self.state}, amount={self.amount:.6f}, model={self.model!r})"


class CostLedger:
    def __init__(self, cap_usd: float | None = None):
        self._cap_usd = cap_usd
        self._settled = 0.0
        self._outstanding = 0.0
        self._refused = 0
        self._lock = threading.Lock()

    # ---------- public accounting API ----------

    def try_reserve(self, amount: float, model: str = "", meta: dict | None = None) -> Reservation | None:
        """Reserve `amount` before a call. Returns None (and counts a refusal)
        when the reservation would push total exposure past the cap."""
        amount = max(0.0, float(amount))
        with self._lock:
            if self._cap_usd is not None and self._settled + self._outstanding + amount > self._cap_usd:
                self._refused += 1
                return None
            self._outstanding += amount
            return Reservation(self, amount, model, meta)

    def complete(self, reservation: Reservation, actual: float | None) -> str:
        """Book the outcome of a finished request. `actual` is the provider-
        reported cost, or None when the response carried no cost (the
        reservation is then retained conservatively). If the reservation was
        abandoned by a local timeout, the actual cost is reconciled against
        the retained estimate; a completed-but-costless abandoned request
        keeps holding its estimate."""
        with self._lock:
            if reservation.state == "abandoned":
                if actual is None:
                    return "retained"
                self._outstanding -= reservation.amount
                self._settled += actual
                reservation.state = "reconciled"
                reservation.actual = actual
                return "reconciled"
            if reservation.state != "reserved":
                return "ignored"
            if actual is None:
                reservation.state = "settled_unknown"
                reservation.actual = None
                return "settled_unknown"
            self._outstanding -= reservation.amount
            self._settled += actual
            reservation.state = "settled"
            reservation.actual = actual
            return "settled"

    def abandon(self, reservation: Reservation) -> str:
        """Retain the reservation after a local timeout: the request may still
        be running and may eventually be billed, so exposure is unchanged."""
        with self._lock:
            if reservation.state == "reserved":
                reservation.state = "abandoned"
                return "abandoned"
            return reservation.state

    def reconcile(self, reservation: Reservation, actual: float) -> str:
        """Book the actual cost of an abandoned request once its worker
        eventually completes. Exposure moves from the retained estimate to the
        actual bill (down or up)."""
        with self._lock:
            if reservation.state == "abandoned":
                self._outstanding -= reservation.amount
                self._settled += actual
                reservation.state = "reconciled"
                reservation.actual = actual
                return "reconciled"
            return "ignored"

    def failure(self, reservation: Reservation) -> str:
        """Request failed with an exception. Active failures released their
        reservation (nothing was produced); abandoned failures retain it (the
        provider may have processed the request before failing)."""
        with self._lock:
            if reservation.state == "reserved":
                self._outstanding -= reservation.amount
                reservation.state = "released"
                return "released"
            return "retained"

    def release(self, reservation: Reservation) -> str:
        with self._lock:
            if reservation.state == "reserved":
                self._outstanding -= reservation.amount
                reservation.state = "released"
                return "released"
            return reservation.state

    # ---------- reporting ----------

    @property
    def settled_usd(self) -> float:
        return self._settled

    @property
    def outstanding_usd(self) -> float:
        return self._outstanding

    @property
    def exposure_usd(self) -> float:
        return self._settled + self._outstanding

    @property
    def cap_usd(self) -> float | None:
        return self._cap_usd

    @property
    def refused_calls(self) -> int:
        return self._refused

    @property
    def tripped(self) -> bool:
        return self._refused > 0 or (
            self._cap_usd is not None and self.exposure_usd >= self._cap_usd
        )

    def status(self) -> dict:
        return {
            "cap_usd": self._cap_usd,
            "settled_usd": round(self._settled, 6),
            "outstanding_usd": round(self._outstanding, 6),
            "exposure_usd": round(self.exposure_usd, 6),
            "tripped": self.tripped,
            "refused_calls": self._refused,
        }

    def __repr__(self) -> str:
        return (
            f"CostLedger(settled={self._settled:.6f}, outstanding={self._outstanding:.6f},"
            f" cap={self._cap_usd})"
        )


def usage_cost(response: Any) -> float | None:
    """Provider-reported cost from a response, or None when absent. Accepts
    either a raw SDK response or an LlmResponse wrapper (unwraps `.raw`)."""
    target = getattr(response, "raw", response)
    usage = getattr(target, "usage", None)
    if usage is None:
        return None
    cost = getattr(usage, "cost", None)
    if isinstance(cost, (int, float)) and cost >= 0:
        return float(cost)
    return None
