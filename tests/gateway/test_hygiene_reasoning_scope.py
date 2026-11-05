"""Gateway hygiene budgets the next reasoning projection before that user row is persisted."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.run_turn_hygiene import GatewayTurnHygieneMixin


def _history():
    return [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "", "reasoning_content": "x" * 100_000,
         "tool_calls": [{"id": "tool", "type": "function",
                         "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "tool", "content": "tool result"},
        {"role": "assistant", "content": "answer", "reasoning_content": "y" * 20_000},
    ]


def _settings(config=None, *, api_mode="chat_completions", provider="deepseek", hard_msg_limit=5000):
    return SimpleNamespace(model="deepseek-flash", provider=provider, base_url="https://api.deepseek.com/v1",
                           api_key="", api_mode=api_mode, data=config or {}, config_context_length=10_000,
                           threshold_pct=.85, hard_msg_limit=hard_msg_limit)


def _runner(monkeypatch, anchor=None):
    monkeypatch.setattr("agent.model_metadata.get_model_context_length_async", AsyncMock(return_value=10_000))
    runner = GatewayTurnHygieneMixin()
    runner._session_has_compression_in_flight = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._session_db._db.get_session_model_config_value.return_value = anchor
    runner._session_db._db.get_compression_failure_cooldown.return_value = None
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("last_prompt", [0, 30_000])
@pytest.mark.parametrize("config", [{}, {"model": {"reasoning_echo": False}},
                                     {"model": {"reasoning_echo": False}, "agent": {"reasoning_echo": True}}])
async def test_next_user_history_reasoning_cannot_trigger_hygiene_or_reuse_old_usage(monkeypatch, last_prompt, config):
    from agent.usage_anchor import capture_usage_anchor

    history = _history()
    original = deepcopy(history)
    legacy = capture_usage_anchor(30_000, 1000, history)
    runner = _runner(monkeypatch, legacy)
    entry = SimpleNamespace(session_id="sid", last_prompt_tokens=last_prompt)

    plan = await runner._hmwa_hygiene_plan(
        _settings(config), history, entry, "key", incoming_message={"role": "user", "content": "hello"},
    )

    assert plan.approx_tokens < 100
    assert not plan.needs_compress
    runner._session_db._db.get_session_model_config_value.assert_not_called()
    assert entry.last_prompt_tokens == last_prompt
    assert history == original


@pytest.mark.asyncio
@pytest.mark.parametrize("config,api_mode,provider", [
    ({"model": {"reasoning_echo": True}}, "chat_completions", "deepseek"),
    ({"agent": {"reasoning_echo": True}}, "chat_completions", "deepseek"),
    ({}, "anthropic_messages", "deepseek"),
    ({}, "codex_responses", "xai"),
    ({}, None, "anthropic"),
])
async def test_native_and_enabled_history_replay_keep_provider_usage_priority(monkeypatch, config, api_mode, provider):
    runner = _runner(monkeypatch)
    entry = SimpleNamespace(session_id="sid", last_prompt_tokens=30_000)

    plan = await runner._hmwa_hygiene_plan(
        _settings(config, api_mode=api_mode, provider=provider), _history(), entry, "key",
        incoming_message={"role": "user", "content": "hello"},
    )

    assert plan.approx_tokens == 30_000
    assert plan.needs_compress


@pytest.mark.asyncio
async def test_internal_notification_keeps_current_tool_loop_reasoning_in_budget(monkeypatch):
    runner = _runner(monkeypatch)
    entry = SimpleNamespace(session_id="sid", last_prompt_tokens=0)

    plan = await runner._hmwa_hygiene_plan(
        _settings(), _history(), entry, "key",
        incoming_message={"role": "user", "content": "background tool notification",
                          "display_kind": "internal_notification"},
    )

    assert plan.approx_tokens >= 30_000
    assert plan.needs_compress


@pytest.mark.asyncio
async def test_reasoning_filter_does_not_disable_the_hard_message_limit(monkeypatch):
    runner = _runner(monkeypatch)
    entry = SimpleNamespace(session_id="sid", last_prompt_tokens=0)

    plan = await runner._hmwa_hygiene_plan(
        _settings(hard_msg_limit=4), _history(), entry, "key",
        incoming_message={"role": "user", "content": "hello"},
    )

    assert plan.approx_tokens < 100
    assert plan.needs_compress
