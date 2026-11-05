"""Reasoning replay is bounded by user turns without deleting durable history."""

from copy import deepcopy

import pytest

from agent.message_sanitization import REASONING_CARRIERS, reasoning_replay_route


def _assistant(label, tool_id=None):
    message = {
        "role": "assistant", "content": label,
        "reasoning_content": f"{label} thought", "reasoning": f"{label} thought",
        "reasoning_details": [{"type": "reasoning.text", "text": f"{label} thought"}],
    }
    if tool_id:
        message["tool_calls"] = [{
            "id": tool_id, "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }]
    return message


def _history():
    return [
        {"role": "user", "content": "previous question"},
        _assistant("old tool", "old"),
        {"role": "tool", "tool_call_id": "old", "content": "previous result"},
        _assistant("old answer"),
        {"role": "user", "content": "current question"},
        _assistant("current first tool", "first"),
        {"role": "tool", "tool_call_id": "first", "content": "first result"},
        _assistant("current second tool", "second"),
        {"role": "tool", "tool_call_id": "second", "content": "second result"},
        _assistant("current answer"),
    ]


def _project(messages, route, **kwargs):
    from agent.message_sanitization import project_reasoning_for_request

    return project_reasoning_for_request(messages, route, **kwargs)


@pytest.mark.parametrize("provider,model,base_url", [
    ("deepseek", "deepseek-flash", "https://api.deepseek.com/v1"),
    ("openrouter", "qwen/qwen3", "https://openrouter.ai/api/v1"),
    ("custom", "qwen3", "http://127.0.0.1:8080/v1"),
])
def test_default_scope_keeps_every_current_iteration_and_strips_previous_turn(provider, model, base_url):
    history = _history()
    original = deepcopy(history)
    route = reasoning_replay_route("chat_completions", provider, model, base_url)

    projected = _project(history, route)

    for index in (1, 3):
        assert not set(REASONING_CARRIERS).intersection(projected[index])
    for index in (5, 7, 9):
        assert projected[index]["reasoning_content"] == history[index]["reasoning_content"]
    assert [m.get("content") for m in projected] == [m.get("content") for m in history]
    assert projected[1]["tool_calls"] == history[1]["tool_calls"]
    assert projected[2] == history[2]
    assert history == original


def test_enabled_scope_replays_previous_reasoning_from_the_untouched_source():
    history = _history()
    original = deepcopy(history)
    disabled = reasoning_replay_route("chat_completions", "openrouter", "qwen/qwen3", "https://openrouter.ai/api/v1")
    enabled = reasoning_replay_route(
        "chat_completions", "openrouter", "qwen/qwen3", "https://openrouter.ai/api/v1", echo_opt_in=True,
    )

    stripped = _project(history, disabled)
    replayed = _project(history, enabled)

    assert "reasoning_content" not in stripped[1]
    for index in (1, 3, 5, 7, 9):
        assert replayed[index]["reasoning_content"] == history[index]["reasoning_content"]
        assert replayed[index]["reasoning_details"] == history[index]["reasoning_details"]
    assert history == original


def test_missing_reasoning_gets_a_pad_only_in_the_current_tool_loop():
    history = _history()
    for message in history:
        for key in REASONING_CARRIERS:
            message.pop(key, None)
    route = reasoning_replay_route("chat_completions", "deepseek", "deepseek-flash", "https://api.deepseek.com/v1")

    projected = _project(history, route)

    for index in (1, 3):
        assert "reasoning_content" not in projected[index]
    assert projected[5]["reasoning_content"] == " "
    assert projected[7]["reasoning_content"] == " "
    assert "reasoning_content" not in history[5]


@pytest.mark.parametrize("nudge", [
    {"role": "user", "content": "continue the existing task", "display_kind": "hidden"},
    pytest.param(None, id="persisted-runtime-nudge"),
])
def test_synthetic_user_rows_do_not_hide_earlier_current_tool_reasoning(nudge):
    from agent.context_compressor import MAX_ITERATIONS_SUMMARY_REQUEST

    history = _history()
    nudge = nudge or {"role": "user", "content": MAX_ITERATIONS_SUMMARY_REQUEST}
    history.insert(7, deepcopy(nudge))
    route = reasoning_replay_route("chat_completions", "deepseek", "deepseek-flash", "https://api.deepseek.com/v1")

    projected = _project(history, route)

    assert "reasoning_content" not in projected[1]
    assert projected[5]["reasoning_content"] == "current first tool thought"
    assert projected[8]["reasoning_content"] == "current second tool thought"


def test_steer_starts_a_new_user_boundary_even_with_an_older_turn_start_index():
    from agent.prompt_builder import steer_user_row

    history = _history()
    history.insert(7, steer_user_row("focus on the second lookup"))
    route = reasoning_replay_route("chat_completions", "deepseek", "deepseek-flash", "https://api.deepseek.com/v1")

    projected = _project(history, route, current_turn_user_idx=4)

    assert "reasoning_content" not in projected[5]
    assert projected[8]["reasoning_content"] == "current second tool thought"
    assert projected[5]["tool_calls"] == history[5]["tool_calls"]
    assert projected[6] == history[6]


def test_history_projection_preserves_private_signature_carriers():
    from agent.reasoning_carriers import CARRIER_TYPE

    record = {"type": CARRIER_TYPE, "extra_content": {"google": {"thought_signature": "SIG"}}}
    history = [
        {"role": "user", "content": "previous question"},
        {**_assistant("old answer"), "reasoning_details": [
            {"type": "reasoning.text", "text": "old answer thought"}, record,
        ]},
        {"role": "user", "content": "current question"},
    ]
    original = deepcopy(history)
    route = reasoning_replay_route(
        "chat_completions", "gemini", "gemini-3.5-pro", "https://generativelanguage.googleapis.com/v1beta/openai",
    )

    projected = _project(history, route)

    assert "reasoning_content" not in projected[1] and "reasoning" not in projected[1]
    assert projected[1]["reasoning_details"] == [record]
    assert history == original


def test_strict_route_never_receives_reasoning_even_when_replay_is_enabled():
    history = _history()
    route = reasoning_replay_route(
        "chat_completions", "mistral", "deepseek-r1", "https://api.mistral.ai/v1", echo_opt_in=True,
    )

    projected = _project(history, route)

    assert not any(set(REASONING_CARRIERS).intersection(m) for m in projected)
    assert projected[1]["tool_calls"] == history[1]["tool_calls"]


def test_native_anthropic_keeps_its_existing_historical_reasoning_contract():
    history = _history()
    route = reasoning_replay_route(
        "anthropic_messages", "custom", "deepseek-flash", "https://api.deepseek.com/anthropic",
    )

    projected = _project(history, route)

    for index in (1, 3, 5, 7, 9):
        assert projected[index]["reasoning_content"] == history[index]["reasoning_content"]
        assert projected[index]["reasoning_details"] == history[index]["reasoning_details"]


def test_a_resumed_tool_chain_without_a_user_anchor_keeps_its_reasoning():
    history = [_assistant("in-flight tool", "pending"),
               {"role": "tool", "tool_call_id": "pending", "content": "result"}]
    route = reasoning_replay_route("chat_completions", "deepseek", "deepseek-flash", "https://api.deepseek.com/v1")

    projected = _project(history, route)

    assert projected[0]["reasoning_content"] == "in-flight tool thought"
    assert projected[1] == history[1]


@pytest.mark.parametrize("current_turn_user_idx", [None, -1, 999, "4", False])
def test_unusable_turn_start_indices_fall_back_to_the_real_user_boundary(current_turn_user_idx):
    history = _history()
    route = reasoning_replay_route("chat_completions", "deepseek", "deepseek-flash", "https://api.deepseek.com/v1")

    projected = _project(history, route, current_turn_user_idx=current_turn_user_idx)

    assert "reasoning_content" not in projected[1]
    assert "reasoning_content" not in projected[3]
    assert projected[5]["reasoning_content"] == "current first tool thought"


def test_private_carrier_cannot_restore_readable_reasoning_on_a_filtered_turn(monkeypatch):
    from agent.reasoning_carriers import CARRIER_TYPE

    monkeypatch.setattr('agent.reasoning_carriers.declared_carrier_keys',
                        lambda: (('opaque_state',), ('provider_reasoning_text',)))
    history = [
        {'role': 'user', 'content': 'old'},
        {'role': 'assistant', 'content': 'answer', 'provider_reasoning_text': 'private thought',
         'reasoning_details': [{'type': CARRIER_TYPE, 'opaque_state': 'SIG',
                                'provider_reasoning_text': 'private thought'}]},
        {'role': 'user', 'content': 'new'},
    ]
    route = reasoning_replay_route('chat_completions', 'custom', 'alias', 'https://example.com/v1')

    projected = _project(history, route)

    assert 'provider_reasoning_text' not in projected[1]
    assert projected[1]['reasoning_details'] == [{'type': CARRIER_TYPE, 'opaque_state': 'SIG'}]
    assert history[1]['provider_reasoning_text'] == 'private thought'
