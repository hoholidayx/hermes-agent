"""Primary request history drops old tool records without losing assistant text or live work."""

from copy import deepcopy
import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_api_request import build_request_payload
from agent.turn_request_assembly import assemble_api_request


def _assistant(text, call_id=None):
    row = {"role": "assistant", "content": text, "reasoning_content": text + " reasoning"}
    if call_id:
        row["tool_calls"] = [{"id": call_id, "type": "function", "function": {
            "name": "lookup", "arguments": "{}",
        }}]
    return row


def _history():
    return [
        {"role": "user", "content": "previous question"},
        _assistant("I will check the old value", "old"),
        {"role": "tool", "tool_call_id": "old", "content": "old secret result"},
        _assistant("The previous answer"),
        {"role": "user", "content": "current question"},
        _assistant("I will check the current value", "current"),
        {"role": "tool", "tool_call_id": "current", "content": "current result"},
    ]


@pytest.fixture
def agent(tmp_path, monkeypatch):
    from run_agent import AIAgent

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: qwen3\n  provider: custom\n  base_url: http://127.0.0.1:8123/v1\n"
        "  context_length: 131072\n  reasoning_echo: false\nagent:\n  tool_history_echo: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        value = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:8123/v1", provider="custom",
            api_mode="chat_completions", model="qwen3", quiet_mode=True,
            skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        )
    value.tools = [{"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {},
    }}}]
    value._use_prompt_caching = False
    value._current_turn_timestamp = time.time()
    value._empty_content_retries = 0
    value._disable_streaming = True
    yield value
    value.close()


def _assemble(agent, history):
    return assemble_api_request(
        agent, messages=history, current_turn_user_idx=4, _ext_prefetch_cache="",
        _plugin_user_context="", moa_config=None, active_system_prompt="system",
        original_user_message="current question", pending_moa_prepared_request=None,
        request_logger=logging.getLogger(__name__),
    )


def _build(agent, assembled, previous=None):
    return build_request_payload(
        agent, api_messages=assembled.api_messages if previous is None else previous.api_messages,
        reasoning_source_messages=assembled.reasoning_source_messages,
        reasoning_replay_key=assembled.reasoning_replay_key if previous is None else previous.reasoning_replay_key,
        _moa_prepared_request=assembled._moa_prepared_request, tools_for_api=assembled.tools_for_api,
        system_message="system", api_call_count=1, api_request_id="request", effective_task_id="task",
        turn_id="turn",
    )


def _calls(messages):
    return [call["id"] for message in messages for call in message.get("tool_calls") or []]


def _assert_filtered(messages):
    assert _calls(messages) == ["current"]
    assert [row["tool_call_id"] for row in messages if row["role"] == "tool"] == ["current"]
    assistant_text = "\n".join(row.get("content") or "" for row in messages if row["role"] == "assistant")
    assert "I will check the old value" in assistant_text
    assert "The previous answer" in assistant_text
    assert not any(row.get("content") == "old secret result" for row in messages)


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("reasoning_echo", [False, True])
def test_actual_payload_filters_history_but_preserves_assistant_text_and_live_tool_loop(agent, bound, reasoning_echo):
    history = _history()
    if bound:
        from agent.tool_history_policy import bind_tool_history_for_turn
        bind_tool_history_for_turn(agent, history)
    agent._reasoning_echo_flag = reasoning_echo
    original = deepcopy(history)

    built = _build(agent, _assemble(agent, history))

    _assert_filtered(built.api_kwargs["messages"])
    assert built.api_kwargs["tools"]
    old_assistant = next(row for row in built.api_kwargs["messages"]
                         if "I will check the old value" in (row.get("content") or ""))
    assert bool(old_assistant.get("reasoning_content")) is reasoning_echo
    if reasoning_echo:
        assert "I will check the old value reasoning" in old_assistant["reasoning_content"]
        assert "The previous answer reasoning" in old_assistant["reasoning_content"]
    assert all("message_uid" not in row for row in built.api_kwargs["messages"])
    assert history == original


def test_selection_and_same_route_retry_cannot_restore_filtered_tools(agent):
    from agent.context_compressor import ContextCompressor
    from agent.tool_history_policy import bind_tool_history_for_turn

    selected = []

    class SelectingCompressor(ContextCompressor):
        def select_context(self, request_messages, *, conversation_messages=None, **kwargs):
            selected.extend(deepcopy(conversation_messages))
            return conversation_messages

    agent.context_compressor = SelectingCompressor(
        model=agent.model, provider=agent.provider, base_url=agent.base_url,
        api_mode=agent.api_mode, config_context_length=131072, quiet_mode=True,
    )
    history = _history()
    bind_tool_history_for_turn(agent, history)
    original = deepcopy(history)

    assembled = _assemble(agent, history)
    first = _build(agent, assembled)
    retried = _build(agent, assembled, previous=first)

    _assert_filtered(first.api_kwargs["messages"])
    _assert_filtered(retried.api_kwargs["messages"])
    assert selected[1]["tool_calls"][0]["id"] == "old"
    assert history == original


@pytest.mark.parametrize("display_kind", ["steer", "hidden"])
def test_request_boundary_survives_display_metadata_cleanup_and_in_loop_user_rows(agent, display_kind):
    from agent.tool_history_policy import bind_tool_history_for_turn

    history = _history()
    bind_tool_history_for_turn(agent, history)
    history += [{"role": "user", "content": "continue this work", "display_kind": display_kind},
                _assistant("second iteration", "second"),
                {"role": "tool", "tool_call_id": "second", "content": "second result"}]
    original = deepcopy(history)

    built = _build(agent, _assemble(agent, history))

    assert _calls(built.api_kwargs["messages"]) == ["current", "second"]
    assert [row["tool_call_id"] for row in built.api_kwargs["messages"] if row["role"] == "tool"] == ["current", "second"]
    assert all("display_kind" not in row for row in built.api_kwargs["messages"])
    assert history == original


def test_supported_fallback_remains_filtered_and_native_fallback_restores_source(agent):
    history = _history()
    assembled = _assemble(agent, history)
    first = _build(agent, assembled)
    _assert_filtered(first.api_messages)

    agent.model = "fallback-model"
    fallback = _build(agent, assembled, previous=first)
    _assert_filtered(fallback.api_messages)

    agent.api_mode = "anthropic_messages"
    # Observe the request presented to the native transport, without calling any provider.
    with patch.object(agent, "_build_api_kwargs", return_value={}):
        native = _build(agent, assembled, previous=fallback)
    assert _calls(native.api_messages) == ["old", "current"]
    assert any(row.get("content") == "old secret result" for row in native.api_messages)


def test_direct_message_builder_filters_a_uid_bound_primary_request(agent):
    from agent.tool_history_policy import bind_tool_history_for_turn
    from agent.turn_context import build_api_messages

    history = _history()
    bind_tool_history_for_turn(agent, history)
    original = deepcopy(history)

    messages, system = build_api_messages(
        agent, history, current_turn_user_idx=4, ext_prefetch_cache="", plugin_user_context="",
        moa_config=None, active_system_prompt="system", preserve_reasoning_sources=False,
    )

    assert system == "system"
    _assert_filtered(messages)
    assert all("message_uid" not in row and "display_kind" not in row for row in messages)
    assert history == original


def test_native_primary_to_supported_fallback_applies_the_requested_filter(agent):
    from agent.tool_history_policy import bind_tool_history_for_turn

    history = _history()
    agent.api_mode = "anthropic_messages"
    bind_tool_history_for_turn(agent, history)
    assembled = _assemble(agent, history)
    with patch.object(agent, "_build_api_kwargs", return_value={}):
        native = _build(agent, assembled)
    assert _calls(native.api_messages) == ["old", "current"]

    agent.api_mode = "chat_completions"
    agent.model = "fallback-model"
    fallback = _build(agent, assembled, previous=native)
    _assert_filtered(fallback.api_kwargs["messages"])


def test_declared_native_client_keeps_tool_history_and_warns_once(agent):
    agent.client.HERMES_SKIP_TRANSPORT_WRAP = True
    agent._emit_warning = MagicMock()
    history = _history()
    original = deepcopy(history)

    assembled = _assemble(agent, history)
    first = _build(agent, assembled)
    retry = _build(agent, assembled, previous=first)

    assert _calls(first.api_kwargs["messages"]) == ["old", "current"]
    assert _calls(retry.api_kwargs["messages"]) == ["old", "current"]
    agent._emit_warning.assert_called_once()
    assert history == original


def test_iteration_limit_summary_uses_the_same_history_filter(agent):
    from agent.chat_completion_helpers import _iteration_summary_api_messages

    history = _history()
    original = deepcopy(history)
    _assert_filtered(_iteration_summary_api_messages(agent, history))
    assert history == original


@pytest.mark.parametrize("stored_content", ["raw unsanitized assistant text", ""])
def test_iteration_summary_retains_the_exact_assistant_sidecar_text_sent_before(agent, stored_content):
    from agent.chat_completion_helpers import _iteration_summary_api_messages

    history = _history()
    history[1]["content"] = stored_content
    history[1]["api_content"] = "I will check the old value"
    original = deepcopy(history)

    messages = _iteration_summary_api_messages(agent, history)

    _assert_filtered(messages)
    assert not any("raw unsanitized assistant text" in (row.get("content") or "") for row in messages)
    assert history == original
