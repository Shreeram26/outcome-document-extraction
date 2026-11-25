"""LLM gateway: the only place that talks to the model provider.

All calls go through the LLM_BASE_URL / LLM_API_KEY environment variables per the
assignment contract; model ids are restricted to the assignment's allow-list.
The OpenAI SDK is imported lazily and the transport is injectable, so tests never
make network calls or spend credits."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

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


class LlmError(Exception):
    """Raised for disallowed models, missing configuration, or failed calls."""


class LlmTimeoutError(LlmError):
    """Raised when a call exceeds the gateway's hard wall-clock deadline."""


@dataclass(frozen=True)
class LlmResponse:
    model: str
    content: str
    tokens_in: int
    tokens_out: int
    raw: Any = None


class LlmGateway:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        wall_clock: float = DEFAULT_WALL_CLOCK_SECONDS,
        transport: Any = None,
    ):
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._wall_clock = wall_clock
        self._transport = transport
        self._client: Any = None

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        if model not in ALLOWED_MODELS:
            raise LlmError(
                f"model {model!r} is not allowed; allowed: {sorted(ALLOWED_MODELS)}"
            )
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self._call_with_watchdog(client, kwargs, model)
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        if content is None:
            raise LlmError(f"model {model} returned no content")
        usage = getattr(response, "usage", None)
        return LlmResponse(
            model=model,
            content=content,
            tokens_in=int(getattr(usage, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "completion_tokens", 0) or 0),
            raw=response,
        )

    def _call_with_watchdog(self, client: Any, kwargs: dict[str, Any], model: str) -> Any:
        """Run the call in a worker thread with a hard wall-clock deadline.
        Provider keep-alives can defeat client read timeouts, so a hung call
        must still become an LlmError instead of blocking forever."""
        outcome: dict[str, Any] = {}

        def run():
            try:
                outcome["response"] = client.chat.completions.create(**kwargs)
            except Exception as exc:
                outcome["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(self._wall_clock)
        if worker.is_alive():
            raise LlmTimeoutError(
                f"LLM call timed out after {self._wall_clock:.0f}s for {model} "
                "(provider kept the connection alive without responding)"
            )
        if "error" in outcome:
            raise LlmError(f"LLM call failed for {model}: {outcome['error']}") from outcome["error"]
        return outcome["response"]

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
