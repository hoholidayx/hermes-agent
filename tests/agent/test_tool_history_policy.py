"""Tool history is a request projection with a stable primary-task boundary."""

from copy import deepcopy
from types import SimpleNamespace

import pytest


def _agent(**overrides):
    values = {
        "api_mode": "chat_completions", "provider": "custom", "model": "test-model",
        "base_url": "http://127.0.0.1:8123/v1", "platform": "cli", "side_agent": False,
    }
    return SimpleNamespace(**(values | overrides))


def _call(call_id, text=""):
    return {"role": "assistant", "content": text, "tool_calls": [{
        "id": call_id, "type": "function", "function": {"name": "lookup", "arguments": "{}"},
    }]}


def _history():
    return [
        {"role": "user", "content": "previous"},
        _call("reused", "old explanation"),
        {"role": "tool", "tool_call_id": "reused", "content": "old result"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current"},
        _call("reused", "live explanation"),
        {"role": "tool", "tool_call_id": "reused", "content": "live result"},
    ]


@pytest.fixture
def disabled_history(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text("agent:\n  tool_history_echo: false\n", encoding="utf-8")
    return config


def test_projection_keeps_text_and_current_reused_call_without_mutating_sources(disabled_history):
    from agent.tool_history_policy import bind_tool_history_for_turn, project_tool_history_for_request

    agent, history = _agent(), _history()
    bind_tool_history_for_turn(agent, history)
    original = deepcopy(history)

    projected = project_tool_history_for_request(agent, history)

    assert projected.source_indices == [0, 1, 3, 4, 5, 6]
    assert projected.messages[1]["content"] == "old explanation"
    assert "tool_calls" not in projected.messages[1]
    assert projected.messages[-2]["tool_calls"][0]["id"] == "reused"
    assert projected.messages[-1]["content"] == "live result"
    projected.messages[-2]["tool_calls"][0]["function"]["arguments"] = "changed"
    assert history == original


@pytest.mark.parametrize("empty_content", ["", "   \n"])
def test_parallel_results_and_empty_call_rows_are_removed_without_losing_multimodal_assistant_text(disabled_history, empty_content):
    from agent.tool_history_policy import bind_tool_history_for_turn, project_tool_history_for_request

    old = _call("old", empty_content)
    old["tool_calls"].append(_call("parallel")["tool_calls"][0])
    visible = _call("visible", [{"type": "text", "text": "assistant explanation"}])
    visible["reasoning_details"] = [{"type": "gemini.native_assistant", "content": "opaque tool replay"}]
    history = [{"role": "user", "content": "old"}, old,
               {"role": "tool", "tool_call_id": "old", "content": "first"},
               {"role": "tool", "tool_call_id": "parallel", "content": [{
                   "type": "image_url", "image_url": {"url": "data:image/png;base64,old-image"},
               }]}, visible, {"role": "tool", "tool_call_id": "visible", "content": "third"},
               {"role": "assistant", "content": "answer"}, {"role": "user", "content": "new"}]
    agent = _agent()
    bind_tool_history_for_turn(agent, history)
    original = deepcopy(history)

    projected = project_tool_history_for_request(agent, history)

    assert projected.source_indices == [0, 4, 6, 7]
    assert projected.messages[1]["content"] == [{"type": "text", "text": "assistant explanation"}]
    assert not projected.messages[1].get("reasoning_details")
    assert not any(row.get("tool_calls") or row["role"] == "tool" for row in projected.messages)
    assert history == original


@pytest.mark.parametrize("suffix", [
    {"role": "user", "content": "adjust current work", "display_kind": "steer"},
    {"role": "user", "content": "continue existing work", "display_kind": "hidden"},
    {"role": "user", "content": "background task completed", "display_kind": "internal_notification"},
])
def test_synthetic_and_steering_rows_do_not_advance_the_tool_boundary(disabled_history, suffix):
    from agent.tool_history_policy import bind_tool_history_for_turn, project_tool_history_for_request

    agent, history = _agent(), _history()
    bind_tool_history_for_turn(agent, history)
    history.extend([suffix, _call("second"), {"role": "tool", "tool_call_id": "second", "content": "second result"}])

    projected = project_tool_history_for_request(agent, history).messages

    assert [row["content"] for row in projected if row["role"] == "tool"] == ["live result", "second result"]


@pytest.mark.parametrize("automatic_resume", [False, True])
def test_resumed_agent_distinguishes_typed_continuation_from_a_new_human_message(disabled_history, automatic_resume):
    from agent.tool_history_policy import bind_tool_history_for_turn, project_tool_history_for_request

    history = _history() + [{"role": "user", "content": "continue the previous work"}]
    if automatic_resume:
        history[-1]["display_kind"] = "internal_notification"
    agent = _agent()
    bind_tool_history_for_turn(agent, history)

    rows = project_tool_history_for_request(agent, history).messages

    assert [row["content"] for row in rows if row["role"] == "tool"] == (["live result"] if automatic_resume else [])


@pytest.mark.parametrize("attributes", [
    {"side_agent": True}, {"platform": "subagent"}, {"_parent_session_id": "parent"},
    {"_delegate_depth": 1}, {"platform": "cron"}, {"platform": "kanban"},
])
def test_non_primary_roles_keep_their_tool_context(disabled_history, attributes):
    from agent.tool_history_policy import bind_tool_history_for_turn, project_tool_history_for_request

    agent, history = _agent(**attributes), _history()
    bind_tool_history_for_turn(agent, history)
    assert project_tool_history_for_request(agent, history).messages == history


def test_delegated_execution_context_is_excluded_but_managed_primary_is_not(disabled_history, monkeypatch):
    from agent.delegation_context import delegated_child_context
    from agent.tool_history_policy import bind_tool_history_for_turn, project_tool_history_for_request

    history = _history()
    with delegated_child_context():
        child = _agent()
        bind_tool_history_for_turn(child, history)
        assert project_tool_history_for_request(child, history).messages == history

    monkeypatch.setattr("agent.runtime_session_store.is_worker_process", lambda: True)
    primary = _agent()
    bind_tool_history_for_turn(primary, history)
    rows = project_tool_history_for_request(primary, history).messages
    assert [row["content"] for row in rows if row["role"] == "tool"] == ["live result"]


@pytest.mark.parametrize("attributes", [
    {"api_mode": "anthropic_messages"}, {"api_mode": "bedrock_converse"},
    {"api_mode": "codex_responses"}, {"api_mode": "codex_app_server"},
    {"provider": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    {"provider": "moa"},
    pytest.param({"client": SimpleNamespace(HERMES_SKIP_TRANSPORT_WRAP=True)}, id="native-client-capability"),
])
def test_native_and_agent_provider_routes_do_not_claim_to_filter_history(disabled_history, attributes):
    from agent.tool_history_policy import bind_tool_history_for_turn, project_tool_history_for_request

    agent, history = _agent(**attributes), _history()
    bind_tool_history_for_turn(agent, history)
    assert project_tool_history_for_request(agent, history).messages == history


def test_normal_new_user_retires_tools_but_missing_boundary_keeps_ambiguous_work(disabled_history):
    from agent.tool_history_policy import bind_tool_history_for_turn, project_tool_history_for_request

    agent, history = _agent(), _history()
    bind_tool_history_for_turn(agent, history)
    history += [{"role": "assistant", "content": "current answer"}, {"role": "user", "content": "next"}]
    bind_tool_history_for_turn(agent, history)
    assert not any(row.get("tool_calls") or row["role"] == "tool"
                   for row in project_tool_history_for_request(agent, history).messages)

    unfinished = [_call("unknown"), {"role": "tool", "tool_call_id": "unknown", "content": "needed"}]
    bind_tool_history_for_turn(agent, unfinished)
    assert project_tool_history_for_request(agent, unfinished).messages == unfinished


def test_setting_is_frozen_inside_a_tool_loop_and_refreshes_at_admission(disabled_history):
    from agent.tool_history_policy import bind_tool_history_for_turn, project_tool_history_for_request

    agent, history = _agent(), _history()
    bind_tool_history_for_turn(agent, history)
    disabled_history.write_text("agent:\n  tool_history_echo: true\n", encoding="utf-8")
    assert [row["content"] for row in project_tool_history_for_request(agent, history).messages
            if row["role"] == "tool"] == ["live result"]
    history += [{"role": "assistant", "content": "answer"}, {"role": "user", "content": "next"}]
    bind_tool_history_for_turn(agent, history)
    assert project_tool_history_for_request(agent, history).messages == history


def test_unsupported_route_warns_once_and_background_agents_do_not_warn(disabled_history):
    from agent.tool_history_policy import warn_unsupported_tool_history

    notices = []
    primary = _agent(api_mode="codex_app_server", _emit_warning=notices.append)
    warn_unsupported_tool_history(primary)
    warn_unsupported_tool_history(primary)
    assert len(notices) == 1 and "retains tool history" in notices[0]

    child = _agent(api_mode="codex_app_server", side_agent=True, _emit_warning=notices.append)
    warn_unsupported_tool_history(child)
    assert len(notices) == 1
