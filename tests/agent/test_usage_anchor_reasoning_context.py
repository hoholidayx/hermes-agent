"""Provider usage is valid only for the reasoning projection the provider priced."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.usage_anchor import (
    anchored_context_tokens,
    capture_usage_anchor,
    persisted_anchor_tokens,
    prepare_reasoning_usage_accounting,
    reasoning_usage_context,
    restore_usage_anchor,
    set_usage_anchor,
)


def _messages():
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "", "reasoning_content": "thinking " * 1000,
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "content": "result", "tool_call_id": "c1"},
    ]


def _agent(replay=False, api_mode="chat_completions", **kwargs):
    return SimpleNamespace(
        provider="deepseek", model="deepseek-flash", base_url="https://api.deepseek.com/v1",
        api_mode=api_mode, _reasoning_echo_flag=replay,
        context_compressor=SimpleNamespace(last_real_prompt_tokens=20_000,
                                           last_compression_rough_tokens=19_000,
                                           awaiting_real_usage_after_compression=True),
        **kwargs,
    )


@pytest.mark.parametrize("replay,api_mode,invalidated", [
    (False, "chat_completions", True),
    (True, "chat_completions", False),
    (False, "anthropic_messages", False),
])
def test_new_user_retires_only_the_current_turn_reasoning_usage(replay, api_mode, invalidated):
    messages = _messages()
    original = deepcopy(messages)
    agent = _agent(replay, api_mode)
    context = reasoning_usage_context(agent, messages)
    anchor = capture_usage_anchor(20_000, 50, messages, reasoning_context=context)
    set_usage_anchor(agent, anchor)

    messages.append({"role": "assistant", "content": "done", "reasoning_content": "final thought"})
    prepare_reasoning_usage_accounting(agent, messages)
    assert anchored_context_tokens(messages, agent._usage_anchor) == 20_050
    assert agent.context_compressor.last_real_prompt_tokens == 20_000

    # Identical user text still starts a distinct turn and retires old thinking.
    messages.append({"role": "user", "content": "hello"})
    prepare_reasoning_usage_accounting(agent, messages)
    if invalidated:
        assert agent._usage_anchor is None
        assert agent.context_compressor.last_real_prompt_tokens == 0
        assert agent.context_compressor.last_compression_rough_tokens == 0
    else:
        assert anchored_context_tokens(messages, agent._usage_anchor) > 20_050
        assert agent.context_compressor.last_real_prompt_tokens == 20_000
    assert agent.context_compressor.awaiting_real_usage_after_compression is True
    assert messages[:len(original)] == original


@pytest.mark.parametrize("change", ["legacy", "model", "flag", "new_user"])
def test_resumed_usage_keeps_projection_provenance_and_rejects_stale_floor(tmp_path, change):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("replay", source="cli")
    messages = _messages()
    live = _agent(session_id="replay", _session_db=db)
    context = None if change == "legacy" else reasoning_usage_context(live, messages)
    set_usage_anchor(live, capture_usage_anchor(20_000, 50, messages, reasoning_context=context))
    fresh = _agent(session_id="replay", _session_db=db)
    restore_usage_anchor(fresh, deepcopy(messages))
    assert fresh._usage_anchor is not None

    if change == "model":
        fresh.model = "different-model"
    elif change == "flag":
        fresh._reasoning_echo_flag = True
    elif change == "new_user":
        messages.extend([{"role": "assistant", "content": "done"}, {"role": "user", "content": "next"}])
        assert persisted_anchor_tokens(db, "replay", messages) is None
    prepare_reasoning_usage_accounting(fresh, messages)
    assert fresh._usage_anchor is None
    assert fresh.context_compressor.last_real_prompt_tokens == 0
    assert fresh.context_compressor.last_compression_rough_tokens == 0
    assert db.get_session_model_config_value("replay", "_usage_anchor") is None
    db.close()
