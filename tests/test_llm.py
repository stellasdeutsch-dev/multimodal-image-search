"""LLM query layer — disabled pass-through behavior (pure; no anthropic needed)."""

from src.llm import QueryAssistant


def test_disabled_passthrough():
    qa = QueryAssistant({"enabled": False})
    assert qa.enabled is False
    assert qa.rewrite("a vague query") == "a vague query"   # unchanged
    assert qa.explain("q", {"path": "x", "score": 0.5}) == ""


def test_enabled_without_key_disables(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    qa = QueryAssistant({"enabled": True})
    assert qa.enabled is False                              # auto-disabled, no key
    assert qa.rewrite("q") == "q"
