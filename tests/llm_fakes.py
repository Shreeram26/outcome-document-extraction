"""Shared stubs for a fake OpenAI-compatible transport. No network, no credits."""
from __future__ import annotations

from types import SimpleNamespace


def fake_response(content: str, tokens_in: int = 100, tokens_out: int = 40, cost: float | None = None):
    usage = SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out)
    if cost is not None:
        usage.cost = cost
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )


class StubCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("no more stub responses configured")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_transport(responses):
    """Returns (transport, completions_stub): transport is an object shaped like
    an OpenAI client (chat.completions.create)."""
    stub = StubCompletions(responses)
    transport = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    return transport, stub
