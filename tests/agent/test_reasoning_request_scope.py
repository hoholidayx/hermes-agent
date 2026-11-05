"""Real agent request assembly keeps replay scope after selection and route changes."""

from copy import deepcopy
import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.message_sanitization import REASONING_CARRIERS
from agent.turn_api_request import build_request_payload
from agent.turn_request_assembly import assemble_api_request, prepare_reasoning_replay_messages


def _assistant(content, call_id=None):
    message = {"role": "assistant", "content": content,
               "reasoning_content": content + " reasoning", "reasoning": content + " reasoning"}
    if call_id:
        message["tool_calls"] = [{"id": call_id, "type": "function",
                                  "function": {"name": "lookup", "arguments": "{}"}}]
    return message


def _history():
    return [
        {"role": "user", "content": "previous question"},
        _assistant("old tool", "old"),
        {"role": "tool", "tool_call_id": "old", "content": "old result"},
        _assistant("old answer"),
        {"role": "user", "content": "current question"},
        _assistant("first current tool", "first"),
        {"role": "tool", "tool_call_id": "first", "content": "first result"},
        _assistant("second current tool", "second"),
        {"role": "tool", "tool_call_id": "second", "content": "second result"},
    ]


@pytest.fixture
def agent(tmp_path, monkeypatch):
    from run_agent import AIAgent

    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: qwen3\n  provider: custom\n  base_url: http://127.0.0.1:8123/v1\n"
        "  context_length: 131072\n  reasoning_echo: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Client construction is the only stub; request phases and transport are real.
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        value = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:8123/v1", provider="custom",
            api_mode="chat_completions", model="qwen3", quiet_mode=True,
            skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        )
    value.tools = [{"type": "function", "function": {
        "name": "lookup", "parameters": {"type": "object", "properties": {}},
    }}]
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
        _moa_prepared_request=assembled._moa_prepared_request,
        tools_for_api=assembled.tools_for_api, system_message="system", api_call_count=1,
        api_request_id="request", effective_task_id="task", turn_id="turn",
    )


def _payload(agent, assembled):
    return _build(agent, assembled).api_kwargs


def _by_content(payload):
    return {message.get("content"): message for message in payload["messages"]
            if isinstance(message.get("content"), str)}


def test_final_request_filters_old_plain_and_tool_reasoning_but_keeps_each_current_iteration(agent):
    history = _history()
    original = deepcopy(history)

    payload = _payload(agent, _assemble(agent, history))

    rows = _by_content(payload)
    assert payload["tools"]
    for content in ("old tool", "old answer"):
        assert not set(REASONING_CARRIERS).intersection(rows[content])
    for content in ("first current tool", "second current tool"):
        assert rows[content]["reasoning_content"] == content + " reasoning"
    assert rows["old tool"]["tool_calls"][0]["id"] == "old"
    assert rows["old result"]["tool_call_id"] == "old"
    assert history == original


def test_retry_with_history_replay_enabled_restores_reasoning_from_assembly_sources(agent):
    history = _history()
    assembled = _assemble(agent, history)
    initial = _payload(agent, assembled)
    assert "reasoning_content" not in _by_content(initial)["old answer"]

    # Route binding supplies this active fallback flag; configuration binding has its own tests.
    agent.model = "fallback-model"
    agent._reasoning_echo_flag = True
    retry = _payload(agent, assembled)

    assert _by_content(retry)["old answer"]["reasoning_content"] == "old answer reasoning"
    assert _by_content(retry)["old tool"]["reasoning_content"] == "old tool reasoning"
    assert "reasoning_content" not in _by_content(initial)["old answer"]
    assert history[3]["reasoning_content"] == "old answer reasoning"


def test_context_selection_cannot_reintroduce_historical_reasoning_into_final_request(agent, monkeypatch):
    history = _history()
    original = deepcopy(history)
    selected = []

    def select_canonical(_agent, _api, canonical, _incoming, logger=None, **kwargs):
        selected.extend(deepcopy(canonical))
        return deepcopy(canonical)

    monkeypatch.setattr("agent.conversation_loop._apply_context_engine_selection", select_canonical)
    payload = _payload(agent, _assemble(agent, history))

    assert selected[1]["reasoning_content"] == "old tool reasoning"
    assert "reasoning_content" not in _by_content(payload)["old tool"]
    assert "reasoning_content" not in _by_content(payload)["old answer"]
    assert _by_content(payload)["first current tool"]["reasoning_content"] == "first current tool reasoning"
    assert history == original


def test_cc_source_rebuilt_for_native_deepseek_keeps_native_pad_contract(agent):
    history = [{"role": "user", "content": "current question"},
               _assistant("pending tool", "pending"),
               {"role": "tool", "tool_call_id": "pending", "content": "result"}]
    history[1].pop("reasoning_content")
    original = deepcopy(history)
    agent.api_mode = "anthropic_messages"
    agent.model = "deepseek-flash"
    agent.base_url = "https://api.deepseek.com/anthropic"

    prepared = prepare_reasoning_replay_messages(agent, history, reconcile_native=True)

    # The legacy native contract pads foreign generic reasoning on tool-call messages.
    assert prepared[1]["reasoning_content"] == " "
    assert "reasoning" not in prepared[1]
    assert history == original


def test_same_route_retry_does_not_restore_rejected_reasoning_details(agent):
    from agent.error_classifier import FailoverReason
    from agent.turn_recovery import _recover_format_errors
    from agent.turn_retry_state import TurnRetryState

    history = _history()
    history[5]["reasoning_details"] = [{"type": "reasoning.text", "text": "bad replay signature"}]
    original = deepcopy(history)
    agent.provider = "openrouter"
    assembled = _assemble(agent, history)
    first = _build(agent, assembled)
    assert _by_content(first.api_kwargs)["first current tool"]["reasoning_details"]

    recovered = _recover_format_errors(
        agent, RuntimeError("invalid thinking signature"),
        SimpleNamespace(reason=FailoverReason.thinking_signature), TurnRetryState(),
        history, first.api_messages,
    )
    assert recovered
    retry = _build(agent, assembled, previous=first)

    assert "reasoning_details" not in _by_content(retry.api_kwargs)["first current tool"]
    assert history == original


def test_final_retry_projection_does_not_treat_hidden_continuation_as_a_new_user(agent):
    history = _history()
    history.insert(7, {'role': 'user', 'content': 'continue the existing task', 'display_kind': 'hidden'})

    payload = _payload(agent, _assemble(agent, history))

    rows = _by_content(payload)
    assert rows['first current tool']['reasoning_content'] == 'first current tool reasoning'
    assert rows['second current tool']['reasoning_content'] == 'second current tool reasoning'
    assert 'reasoning_content' not in rows['old tool']
    assert all('display_kind' not in message for message in payload['messages'])


def test_final_carrier_reconciliation_preserves_only_current_required_pads(agent):
    history = _history()
    agent.model = 'deepseek-flash'
    for message in history:
        for key in REASONING_CARRIERS:
            message.pop(key, None)

    payload = _payload(agent, _assemble(agent, history))

    rows = _by_content(payload)
    assert 'reasoning_content' not in rows['old tool']
    assert rows['first current tool']['reasoning_content'] == ' '
    assert rows['second current tool']['reasoning_content'] == ' '
