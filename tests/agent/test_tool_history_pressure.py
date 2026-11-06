"""Compression pressure prices the primary session's tool replay projection."""

from copy import deepcopy
from types import SimpleNamespace

from agent.context_compressor_tail import tail_row_pricer
from agent.turn_context import _preflight_request_tokens
from agent.usage_anchor import (
    anchored_context_tokens,
    capture_usage_anchor,
    persisted_anchor_tokens,
    prepare_reasoning_usage_accounting,
    reasoning_usage_context,
    restore_usage_anchor,
    set_usage_anchor,
)


def _agent(**overrides):
    fields = dict(
        api_mode="chat_completions", provider="custom", model="test-model",
        base_url="http://127.0.0.1:8123/v1", _reasoning_echo_flag=True,
        _tool_history_echo_flag=False, _tool_history_primary=True, tools=[],
        context_compressor=SimpleNamespace(
            api_mode="chat_completions", provider="custom", model="test-model",
            base_url="http://127.0.0.1:8123/v1",
            last_real_prompt_tokens=0, last_compression_rough_tokens=0,
            awaiting_real_usage_after_compression=True,
        ),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _call(call_id, text=""):
    return {"role": "assistant", "content": text, "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": "lookup", "arguments": '{"query":"' + "x" * 80_000 + '"}'},
    }]}


def _messages():
    return [
        {"role": "user", "content": "previous"}, _call("old", "Let me look."),
        {"role": "tool", "tool_call_id": "old", "content": "y" * 200_000},
        {"role": "assistant", "content": "previous answer"},
        {"role": "user", "content": "current"}, _call("first"),
        {"role": "tool", "tool_call_id": "first", "content": "z" * 40_000},
    ]


def test_preflight_and_tail_ignore_old_tool_payload_but_keep_active_loop():
    agent = _agent()
    del agent._tool_history_primary  # Direct preflight callers have not admitted a turn yet.
    messages = _messages()
    original = deepcopy(messages)

    estimate = _preflight_request_tokens(agent, messages, "")
    price = tail_row_pricer(agent.context_compressor, messages)

    assert 30_000 <= estimate < 31_000
    assert 0 < price(1) < 100  # Assistant narration survives without call arguments.
    assert price(2) == 0
    assert price(5) >= 20_000
    assert price(6) >= 10_000
    assert messages == original


def test_client_that_keeps_tool_history_keeps_it_in_compression_tail_budget():
    agent = _agent(client=SimpleNamespace(SUPPORTS_HERMES_TOOL_CALLS=False))
    messages = _messages()

    estimate = _preflight_request_tokens(agent, messages, "")
    price = tail_row_pricer(agent.context_compressor, messages)

    assert estimate > 100_000
    assert price(1) >= 20_000
    assert price(2) >= 50_000


def test_new_user_retires_tool_usage_even_when_reasoning_history_is_enabled():
    agent = _agent()
    messages = _messages()
    anchor = capture_usage_anchor(100_000, 100, messages, reasoning_context=reasoning_usage_context(agent, messages))
    set_usage_anchor(agent, anchor)
    agent.context_compressor.last_real_prompt_tokens = 100_000
    agent.context_compressor.last_compression_rough_tokens = 99_000
    messages.append({"role": "assistant", "content": "done"})
    prepare_reasoning_usage_accounting(agent, messages)
    assert anchored_context_tokens(messages, agent._usage_anchor) == 100_100

    messages.append({"role": "user", "content": "next question"})
    assert anchored_context_tokens(messages, anchor) is None
    estimate = _preflight_request_tokens(agent, messages, "")

    assert estimate < 500
    assert agent._usage_anchor is None
    assert agent.context_compressor.last_real_prompt_tokens == 0
    assert agent.context_compressor.last_compression_rough_tokens == 0
    assert agent.context_compressor.awaiting_real_usage_after_compression is True


def test_enabling_tool_filter_invalidates_legacy_usage_and_disabling_it_restores_pressure():
    agent = _agent(_tool_history_echo_flag=True)
    messages = _messages()
    context = reasoning_usage_context(agent, messages)
    set_usage_anchor(agent, capture_usage_anchor(100_000, 100, messages, reasoning_context=context))

    agent._tool_history_echo_flag = False
    prepare_reasoning_usage_accounting(agent, messages)

    assert agent._usage_anchor is None
    assert _preflight_request_tokens(agent, messages, "") < 31_000
    agent._tool_history_echo_flag = True
    assert _preflight_request_tokens(agent, messages, "") > 100_000


def test_restored_anchor_keeps_tool_projection_and_rejects_next_user(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("tools", source="cli")
    messages = _messages()
    live = _agent(session_id="tools", _session_db=db)
    set_usage_anchor(live, capture_usage_anchor(
        30_000, 100, messages, reasoning_context=reasoning_usage_context(live, messages),
    ))
    fresh = _agent(session_id="tools", _session_db=db)
    restore_usage_anchor(fresh, deepcopy(messages))
    prepare_reasoning_usage_accounting(fresh, messages)
    assert anchored_context_tokens(messages, fresh._usage_anchor) == 30_100

    messages.extend([{"role": "assistant", "content": "done"}, {"role": "user", "content": "next"}])
    assert persisted_anchor_tokens(db, "tools", messages) is None
    prepare_reasoning_usage_accounting(fresh, messages)
    assert fresh._usage_anchor is None
    assert db.get_session_model_config_value("tools", "_usage_anchor") is None
    db.close()
