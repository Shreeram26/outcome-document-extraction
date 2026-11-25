from __future__ import annotations

import pytest

from app.extraction import ALLOWED_MODELS, LlmError, LlmGateway
from llm_fakes import fake_response, make_transport

MESSAGES = [{"role": "user", "content": "extract"}]


def test_complete_returns_content_and_usage():
    transport, stub = make_transport([fake_response("hi", tokens_in=11, tokens_out=3)])
    gateway = LlmGateway(transport=transport)
    response = gateway.complete("z-ai/glm-5.3-flash", MESSAGES)
    assert response.content == "hi"
    assert response.tokens_in == 11
    assert response.tokens_out == 3
    assert response.model == "z-ai/glm-5.3-flash"
    assert stub.calls[0]["model"] == "z-ai/glm-5.3-flash"
    assert stub.calls[0]["messages"] == MESSAGES
    assert stub.calls[0]["temperature"] == 0.0


def test_max_tokens_forwarded_when_given():
    transport, stub = make_transport([fake_response("hi")])
    gateway = LlmGateway(transport=transport)
    gateway.complete("google/gemini-3.8-flash", MESSAGES, max_tokens=500)
    assert stub.calls[0]["max_tokens"] == 500


def test_disallowed_model_rejected_without_any_call():
    transport, stub = make_transport([fake_response("hi")])
    gateway = LlmGateway(transport=transport)
    with pytest.raises(LlmError, match="not allowed"):
        gateway.complete("gpt-4o", MESSAGES)
    assert stub.calls == []


def test_allowlist_matches_assignment_section_3():
    assert ALLOWED_MODELS == frozenset(
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


def test_transport_failure_is_wrapped_in_llm_error():
    transport, _ = make_transport([RuntimeError("connection boom")])
    gateway = LlmGateway(transport=transport)
    with pytest.raises(LlmError, match="connection boom"):
        gateway.complete("google/gemini-3.8-flash", MESSAGES)


def test_empty_response_rejected():
    transport, _ = make_transport([fake_response(None)])
    gateway = LlmGateway(transport=transport)
    with pytest.raises(LlmError, match="no content"):
        gateway.complete("google/gemini-3.8-flash", MESSAGES)


def test_missing_env_variables_raises(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    gateway = LlmGateway()
    with pytest.raises(LlmError, match="LLM_BASE_URL"):
        gateway.complete("google/gemini-3.8-flash", MESSAGES)


def test_missing_api_key_only_raises_for_that_var(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    gateway = LlmGateway()
    with pytest.raises(LlmError, match="LLM_API_KEY"):
        gateway.complete("google/gemini-3.8-flash", MESSAGES)
