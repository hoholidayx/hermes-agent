"""Reasoning replay configuration resolves from real profile files and freezes per user turn."""

from types import SimpleNamespace

import pytest

from agent.route_binding import reasoning_echo_for_fallback
from run_agent import AIAgent


@pytest.mark.parametrize("config,expected", [
    ("model: test-model\n", False),
    ("model:\n  reasoning_echo: false\n", False),
    ("model:\n  reasoning_echo: true\n", True),
    ("agent:\n  reasoning_echo: true\n", True),
    ("agent:\n  reasoning_echo: false\n", False),
    ("model:\n  reasoning_echo: false\nagent:\n  reasoning_echo: true\n", False),
    ("model:\n  reasoning_echo: true\nagent:\n  reasoning_echo: false\n", True),
    ("model:\n  reasoning_echo: 'false'\n", False),
])
def test_profile_config_resolves_model_precedence_and_agent_alias(tmp_path, monkeypatch, config, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(config, encoding="utf-8")

    agent = object.__new__(AIAgent)
    assert agent._read_reasoning_echo_from_config() is expected


def test_primary_setting_refreshes_only_at_next_user_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  reasoning_echo: true\n", encoding="utf-8")
    agent = object.__new__(AIAgent)
    agent.api_mode = "chat_completions"
    agent._primary_runtime = {"reasoning_echo_flag": False}

    agent._refresh_reasoning_echo_for_turn()
    assert agent._reasoning_echo_opt_in() is True
    assert agent._primary_runtime["reasoning_echo_flag"] is True

    config.write_text("model:\n  reasoning_echo: false\n", encoding="utf-8")
    assert agent._reasoning_echo_opt_in() is True
    agent._refresh_reasoning_echo_for_turn()
    assert agent._reasoning_echo_opt_in() is False
    assert agent._primary_runtime["reasoning_echo_flag"] is False


@pytest.mark.parametrize("primary,entry,expected", [
    (False, {}, False),
    (True, {}, True),
    (True, {"reasoning_echo": False}, False),
    (False, {"reasoning_echo": True}, True),
])
def test_fallback_inherits_primary_instead_of_previous_route(primary, entry, expected):
    agent = SimpleNamespace(api_mode="chat_completions", _reasoning_echo_flag=not primary,
                            _primary_runtime={"reasoning_echo_flag": primary})

    assert reasoning_echo_for_fallback(agent, entry) is expected


def test_primary_restore_recovers_refreshed_setting(tmp_path, monkeypatch):
    from agent.agent_runtime_helpers import _apply_primary_runtime_fields

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  reasoning_echo: true\n", encoding="utf-8")
    agent = object.__new__(AIAgent)
    agent.api_mode = "chat_completions"
    agent._primary_runtime = {
        "model": "primary", "provider": "custom", "base_url": "http://127.0.0.1:8098/v1",
        "api_mode": "chat_completions", "api_key": "test-key", "client_kwargs": {},
    }
    agent._refresh_reasoning_echo_for_turn()
    agent._reasoning_echo_flag = False

    _apply_primary_runtime_fields(agent, agent._primary_runtime)

    assert agent._reasoning_echo_opt_in() is True


@pytest.mark.parametrize("api_mode", ["anthropic_messages", "codex_responses"])
def test_native_routes_keep_existing_fallback_and_refresh_behavior(tmp_path, monkeypatch, api_mode):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  reasoning_echo: false\n", encoding="utf-8")
    agent = object.__new__(AIAgent)
    agent.api_mode, agent._reasoning_echo_flag = api_mode, True
    agent._primary_runtime = {"reasoning_echo_flag": True}

    agent._refresh_reasoning_echo_for_turn()

    assert agent._reasoning_echo_flag is True
    assert reasoning_echo_for_fallback(agent, {}) is False
    assert reasoning_echo_for_fallback(agent, {"reasoning_echo": True}) is True


@pytest.mark.parametrize("api_mode", ["anthropic_messages", "codex_responses"])
def test_agent_alias_does_not_change_native_initial_setting(tmp_path, monkeypatch, api_mode):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    config.write_text("agent:\n  reasoning_echo: true\n", encoding="utf-8")
    agent = object.__new__(AIAgent)
    agent.api_mode = api_mode
    assert agent._read_reasoning_echo_from_config() is False

    config.write_text("model:\n  reasoning_echo: true\n", encoding="utf-8")
    assert agent._read_reasoning_echo_from_config() is True


@pytest.mark.parametrize("resolved_mode,expected", [("chat_completions", True), ("anthropic_messages", False)])
def test_model_switch_reads_setting_for_resolved_transport(tmp_path, monkeypatch, resolved_mode, expected):
    from agent import agent_runtime_helpers as runtime

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("agent:\n  reasoning_echo: true\n", encoding="utf-8")
    agent = object.__new__(AIAgent)
    agent.api_mode = "codex_responses"
    agent._credential_pool = object()
    monkeypatch.setattr(runtime, "_build_switched_client", lambda *a: setattr(agent, "api_mode", resolved_mode))
    monkeypatch.setattr(runtime, "sync_credential_pool_entry_id", lambda *a: None)

    runtime._swap_switch_runtime(agent, "next", "custom", "key", "http://127.0.0.1:8098/v1",
                                 "codex_responses", "custom", "custom", "custom")

    assert agent._reasoning_echo_flag is expected
