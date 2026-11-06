"""Tool-history replay settings come from the owning profile through the real config loader."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("agent_config", "expected"),
    [({}, True), ({"tool_history_echo": True}, True), ({"tool_history_echo": False}, False),
     ({"tool_history_echo": "false"}, True), ({"tool_history_echo": None}, True),
     ({"tool_history_echo": 0}, True)],
    ids=["default", "enabled", "disabled", "string", "null", "integer"],
)
def test_only_a_yaml_boolean_false_disables_tool_history(tmp_path, monkeypatch, agent_config, expected):
    from agent.tool_history_policy import read_tool_history_echo
    from hermes_cli.config import atomic_config_write, load_config_readonly

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {"agent": agent_config})

    assert load_config_readonly()["agent"].get("tool_history_echo", True) == agent_config.get("tool_history_echo", True)
    assert read_tool_history_echo(SimpleNamespace()) is expected


def test_tool_history_setting_uses_the_bound_profile_and_restores_the_launch_profile(tmp_path, monkeypatch):
    from agent.tool_history_policy import read_tool_history_echo
    from hermes_cli.config import atomic_config_write
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch, routed = tmp_path / "launch", tmp_path / "routed"
    launch.mkdir()
    routed.mkdir()
    atomic_config_write(launch / "config.yaml", {"agent": {"tool_history_echo": True}})
    atomic_config_write(routed / "config.yaml", {"agent": {"tool_history_echo": False}})
    monkeypatch.setenv("HERMES_HOME", str(launch))
    agent = SimpleNamespace()

    assert read_tool_history_echo(agent) is True
    token = set_hermes_home_override(routed)
    try:
        assert read_tool_history_echo(agent) is False
    finally:
        reset_hermes_home_override(token)
    assert read_tool_history_echo(agent) is True


@pytest.mark.parametrize("following", [False, True], ids=["frozen", "following"])
def test_local_policy_follows_tool_history_only_when_its_configuration_follows_the_profile(tmp_path, monkeypatch, following):
    from agent.tool_history_policy import read_tool_history_echo
    from gateway.session_policy import build_policy, policy_scope
    from hermes_cli.config import atomic_config_write
    from hermes_cli.config_effective import load_user_config_effective

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    config = {"model": {"default": "test-main", "provider": "custom", "base_url": "http://127.0.0.1:9/v1"},
              "agent": {"tool_history_echo": True}, "platform_toolsets": {"cli": ["terminal"]}}
    atomic_config_write(path, config)
    policy = build_policy({"source": "cli", "cwd": str(tmp_path), "follow_profile_config": following},
                          load_user_config_effective(path))
    authority = SimpleNamespace(db=SimpleNamespace(db_path=str(tmp_path / "state.db")))

    atomic_config_write(path, {**config, "agent": {"tool_history_echo": False}})

    assert policy.config(authority, secrets=False)["agent"]["tool_history_echo"] is not following
    with policy_scope(policy, authority=authority):
        assert read_tool_history_echo(SimpleNamespace()) is not following
