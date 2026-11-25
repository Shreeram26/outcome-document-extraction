"""LLM gateway: the only place that talks to the model provider.

All calls go through the LLM_BASE_URL / LLM_API_KEY environment variables per the
assignment contract; model ids are restricted to the assignment's allow-list.
The OpenAI SDK is imported lazily and the transport is injectable, so tests never
make network calls or spend credits.

Cost control: when a CostLedger is attached, every call reserves an estimated
maximum cost BEFORE it is launched (input tokens estimated from message sizes,
output bounded by an enforced max_tokens, priced per model and multiplied by a
safety factor). The reservation is settled with the provider-reported cost on
completion, retained when the call is abandoned by the watchdog (the provider
may still bill it), reconciled when an abandoned request eventually completes,
and released when a request fails without producing a response. A reservation
that would push total exposure past the cap is refused before any network use."""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Any

from .ledger import CostLedger, Reservation, usage_cost

ALLOWED_MODELS = frozenset(
    {
        "google/gemini-3.8-flash",
        "google/gemini-3.1-pro-preview",
        "anthropic/claude-haiku-5.5",
        "anthropic/claude-sonnet-5.5",
        "openai/gpt-6-luna",
        "openai/gpt-6.1-sol",
        "z-ai/glm-5.3-flash",
        "qwen/qwen3-vl-235b-a22b-instruct",
        "deepseek/deepseek-v3.2",
    }
)

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_WALL_CLOCK_SECONDS = 150.0
DEFAULT_MAX_TOKENS = 16000
DEFAULT_SAFETY_FACTOR = 1.25

# Budgeting estimates only (USD per 1M tokens); settlement always uses the
# provider-reported usage.cost when present. Conservative for unknown models.
PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "google/gemini-3.8-flash": {"input": 0.50, "output": 2.50},
    "google/gemini-3.1-pro-preview": {"input": 5.0, "output": 25.0},
    "anthropic/claude-haiku-5.5": {"input": 1.0, "output": 5.0},
    "anthropic/claude-sonnet-5.5": {"input": 3.0, "output": 15.0},
    "openai/gpt-6-luna": {"input": 0.75, "output": 4.0},
    "openai/gpt-6.1-sol": {"input": 4.0, "output": 20.0},
    "z-ai/glm-5.3-flash": {"input": 0.30, "output": 1.50},
    "qwen/qwen3-vl-235b-a22b-instruct": {"input": 0.60, "output": 3.0},
    "deepseek/deepseek-v3.2": {"input": 0.40, "output": 2.0},
    "default": {"input": 3.0, "output": 15.0},
}


class LlmError(Exception):
    """Raised for disallowed models, missing configuration, or failed calls."""


class LlmTimeoutError(LlmError):
    """Raised when a call exceeds the gateway's hard wall-clock deadline."""


class LlmBudgetError(LlmError):
    """Raised when the cost ledger refuses a reservation (cap would be exceeded)."""


class LlmBadResponseError(LlmError):
    """Raised when a response arrived but is unusable (e.g. no content). The
    request may already have incurred provider cost: `actual_cost_usd` carries
    the provider-reported cost when known, so the caller must not refund the
    reservation blindly."""

    def __init__(self, message: str, actual_cost_usd: float | None = None):
        super().__init__(message)
        self.actual_cost_usd = actual_cost_usd
        self.response_arrived = True


@dataclass(frozen=True)
class LlmResponse:
    model: str
    content: str
    tokens_in: int
    tokens_out: int
    finish_reason: str | None = None
    raw: Any = None


class LlmGateway:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        wall_clock: float = DEFAULT_WALL_CLOCK_SECONDS,
        transport: Any = None,
        ledger: CostLedger | None = None,
        default_max_tokens: int | None = DEFAULT_MAX_TOKENS,
        pricing: dict[str, dict[str, float]] | None = None,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
    ):
        if default_max_tokens is None or default_max_tokens <= 0:
            raise ValueError(
                "default_max_tokens must be a positive integer; an unbounded or"
                " non-positive output limit makes cost reservations meaningless"
            )
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._wall_clock = wall_clock
        self._transport = transport
        self._client: Any = None
        self.ledger = ledger
        self.default_max_tokens = default_max_tokens
        self._pricing = pricing or PRICING_USD_PER_MTOK
        self.safety_factor = safety_factor

    def estimate_cost_usd(
        self, messages: list[dict[str, Any]], model: str, max_tokens: int | None = None
    ) -> float:
        """Conservative pre-call cost estimate: input tokens from message sizes
        (including multimodal image blocks, estimated at 3000 tokens each),
        output tokens bounded by the enforced max_tokens, priced per model and
        multiplied by the safety factor."""
        effective = max_tokens if max_tokens is not None else self.default_max_tokens
        in_tokens = 32
        image_count = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                in_tokens += math.ceil(len(content) / 4)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            in_tokens += math.ceil(len(block.get("text", "")) / 4)
                        elif block.get("type") == "image_url":
                            image_count += 1
        in_tokens += image_count * 3000
        out_tokens = effective or 0
        price = self._pricing.get(model, self._pricing.get("default", {"input": 3.0, "output": 15.0}))
        estimate = (in_tokens * price["input"] + out_tokens * price["output"]) / 1_000_000
        return estimate * self.safety_factor

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> LlmResponse:
        if model not in ALLOWED_MODELS:
            raise LlmError(
                f"model {model!r} is not allowed; allowed: {sorted(ALLOWED_MODELS)}"
            )
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer when provided")
        client = self._get_client()
        effective_max = max_tokens if max_tokens is not None else self.default_max_tokens
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": effective_max,
        }
        if extra_params:
            kwargs.update(extra_params)
        reservation: Reservation | None = None
        if self.ledger is not None:
            estimate = self.estimate_cost_usd(messages, model, effective_max)
            reservation = self.ledger.try_reserve(estimate, model=model, meta={"max_tokens": effective_max})
            if reservation is None:
                raise LlmBudgetError(
                    "cost budget exhausted: reservation refused"
                    f" (est ${estimate:.4f}, exposure ${self.ledger.exposure_usd:.4f}"
                    f" >= cap ${self.ledger.cap_usd:.2f})"
                )
        response = self._call_with_watchdog(client, kwargs, model, reservation)
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        if content is None:
            raise LlmBadResponseError(
                self._describe_empty_response(model, choice, response),
                actual_cost_usd=usage_cost(response),
            )
        usage = getattr(response, "usage", None)
        return LlmResponse(
            model=model,
            content=content,
            tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=getattr(choice, "finish_reason", None),
            raw=response,
        )

    def _call_with_watchdog(
        self, client: Any, kwargs: dict[str, Any], model: str, reservation: Reservation | None
    ) -> Any:
        """Run the call in a worker thread with a hard wall-clock deadline.
        Provider keep-alives can defeat client read timeouts, so a hung call
        must still become an LlmTimeoutError instead of blocking forever.

        The worker reconciles its own reservation: a normal completion settles
        it with the provider-reported cost; a completion arriving after the
        watchdog abandoned it reconciles the retained estimate against the
        actual bill."""
        outcome: dict[str, Any] = {}

        def run():
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                outcome["error"] = exc
                if reservation is not None:
                    outcome["reservation"] = self.ledger.failure(reservation)
                return
            if reservation is not None:
                booked = self.ledger.complete(reservation, usage_cost(response))
                if booked in ("reconciled", "retained"):
                    return
            outcome["response"] = response

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(self._wall_clock)
        if "response" in outcome:
            return outcome["response"]
        if "error" in outcome:
            raise LlmError(f"LLM call failed for {model}: {outcome['error']}") from outcome["error"]
        if worker.is_alive():
            if reservation is not None:
                self.ledger.abandon(reservation)
            if "response" in outcome:
                return outcome["response"]
            raise LlmTimeoutError(
                f"LLM call timed out after {self._wall_clock:.0f}s for {model} "
                "(provider kept the connection alive without responding;"
                " reservation retained until the request settles)"
            )
        raise LlmError(f"LLM worker for {model} finished without a result")

    def _describe_empty_response(self, model: str, choice: Any, response: Any) -> str:
        """Diagnostic detail for responses with no content: finish reason,
        non-standard message fields (e.g. provider-specific reasoning), and
        whether usage/cost were reported."""
        finish = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        extra_keys: list[str] = []
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            extra_keys = sorted(extra)
        else:
            extra_keys = sorted(
                k for k in dir(message) if not k.startswith("_") and getattr(message, k, None) is not None
            )[:8]
        usage = getattr(response, "usage", None)
        has_cost = getattr(usage, "cost", None) is not None
        return (
            f"model {model} returned no content (finish_reason={finish!r},"
            f" message_fields={extra_keys}, usage_cost={has_cost})"
        )

    def _get_client(self) -> Any:
        if self._transport is not None:
            return self._transport
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        from openai import OpenAI

        base_url = self._base_url or os.environ.get("LLM_BASE_URL")
        api_key = self._api_key or os.environ.get("LLM_API_KEY")
        missing = [
            name
            for name, value in (("LLM_BASE_URL", base_url), ("LLM_API_KEY", api_key))
            if not value
        ]
        if missing:
            raise LlmError(f"missing environment variables: {', '.join(missing)}")
        return OpenAI(base_url=base_url, api_key=api_key, timeout=self._timeout)
