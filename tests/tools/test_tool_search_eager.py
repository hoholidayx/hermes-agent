"""Selected plugin tools stay direct without changing session grants or frozen routing."""
import json
from types import SimpleNamespace

import pytest

from tools.tool_search import ToolSearchConfig, load_config_readonly
from tools.tool_search_policy import agent_config, bind_agent_config


IMAGE = "eager_probe_image"
ADMIN = "eager_probe_admin"
DISABLED = "eager_probe_unavailable"
TOOLSET = "eager-probe"


@pytest.fixture
def configured(tmp_path, monkeypatch):
    from hermes_cli.config import atomic_config_write
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def save(raw):
        atomic_config_write(tmp_path / "config.yaml", {"tools": {"tool_search": raw}})
        return load_config_readonly()

    return save


@pytest.fixture
def registered():
    from tools.registry import registry
    for name in (IMAGE, ADMIN, DISABLED):
        registry.register(
            name=name, toolset=TOOLSET,
            schema={"name": name, "description": "Probe image capability.",
                    "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kwargs: json.dumps({"success": True}),
            check_fn=(lambda: False) if name == DISABLED else None,
        )
    yield registry
    for name in (IMAGE, ADMIN, DISABLED):
        registry.deregister(name)


def _names(definitions):
    return {row["function"]["name"] for row in definitions}


def _agent(config, *, session_id="eager-session"):
    import model_tools
    agent = SimpleNamespace(
        _tool_search_config=config, enabled_toolsets=[TOOLSET], disabled_toolsets=[],
        platform="cli", session_id=session_id, tools=[], valid_tool_names=set(),
    )
    with bind_agent_config(agent):
        agent.tools = model_tools.get_tool_definitions(enabled_toolsets=[TOOLSET], quiet_mode=True)
    agent.valid_tool_names = _names(agent.tools)
    return agent


def test_eager_config_round_trip_and_conflict(configured, caplog):
    config = configured({"eager": [IMAGE, "", IMAGE], "defer": [IMAGE, ADMIN]})
    assert config.eager_tools == {IMAGE}
    assert ToolSearchConfig.from_raw(config.as_raw()) == config
    assert "overrides defer" in caplog.text
    assert ToolSearchConfig.from_raw({"eager": IMAGE}).eager_tools == set()
    assert ToolSearchConfig.from_raw({"eager": [None, 1]}).eager_tools == set()
    assert ToolSearchConfig.from_raw(None).eager_tools == set()


def test_one_plugin_stays_direct_and_bridge_routes_consistently(configured, registered):
    import model_tools
    from tools.tool_search import dispatch_tool_search, dispatch_tool_describe, scoped_deferrable_names
    config = configured({"enabled": "on", "eager": [IMAGE], "defer": [IMAGE, ADMIN]})
    agent = _agent(config)
    assert IMAGE in agent.valid_tool_names
    assert ADMIN not in agent.valid_tool_names
    assert "tool_call" in agent.valid_tool_names
    with bind_agent_config(agent):
        raw = model_tools.get_tool_definitions(enabled_toolsets=[TOOLSET], quiet_mode=True,
                                               skip_tool_search_assembly=True)
        assert IMAGE not in scoped_deferrable_names(raw)
        assert ADMIN in scoped_deferrable_names(raw)
        search = json.loads(dispatch_tool_search({"queries": ["probe"]}, current_tool_defs=raw))
        assert ADMIN in search["tools"] and IMAGE not in search["tools"]
        described = json.loads(dispatch_tool_describe({"names": [IMAGE, ADMIN]}, current_tool_defs=raw))
        assert ADMIN in described["tools"] and IMAGE in described["errors"]
        assert json.loads(model_tools.handle_function_call(IMAGE, {}))["success"]
        result = model_tools.handle_function_call("tool_call", {"calls": [{"name": ADMIN, "arguments": {}}]},
                                                   enabled_toolsets=[TOOLSET])
        assert json.loads(result)["success"]
        direct = model_tools.handle_function_call("tool_call", {"calls": [{"name": IMAGE, "arguments": {}}]},
                                                   enabled_toolsets=[TOOLSET])
        assert "direct" in json.loads(direct)["error"].lower()


def test_eager_never_adds_disabled_unavailable_unknown_or_prefix_matches(configured, registered):
    import model_tools
    configured({"eager": [IMAGE, DISABLED, "not_installed", "eager_probe_*"]})
    allowed = model_tools.get_tool_definitions(enabled_toolsets=[TOOLSET], quiet_mode=True)
    assert IMAGE in _names(allowed) and DISABLED not in _names(allowed)
    assert ADMIN not in _names(allowed)
    assert "not_installed" not in _names(allowed)
    denied = model_tools.get_tool_definitions(enabled_toolsets=[], disabled_toolsets=[TOOLSET], quiet_mode=True)
    assert not ({IMAGE, ADMIN, DISABLED} & _names(denied))
    result = model_tools.handle_function_call("tool_call", {"calls": [{"name": ADMIN, "arguments": {}}]},
                                               enabled_toolsets=[], disabled_toolsets=[TOOLSET])
    assert "error" in json.loads(result)


def test_saved_policy_survives_config_changes_and_restore(configured, registered, tmp_path):
    from hermes_state import SessionDB
    from tools.mcp_tool_agent import persist_agent_tool_names, restore_agent_tool_prefix, refresh_agent_mcp_tools
    old = _agent(configured({"eager": [IMAGE]}))
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session(old.session_id, source="cli")
        old._session_db = db
        persist_agent_tool_names(old)
        saved = json.loads(db.get_session(old.session_id)["tool_names"])
        assert saved["tool_search"]["eager"] == [IMAGE]
        fresh = _agent(configured({"eager": [ADMIN]}))
        assert ADMIN in fresh.valid_tool_names and IMAGE not in fresh.valid_tool_names
        fresh._session_db = db
        restore_agent_tool_prefix(fresh, saved)
        assert fresh.tools == saved["tools"]
        assert agent_config(fresh).eager_tools == {IMAGE}
        before = json.dumps(fresh.tools)
        refresh_agent_mcp_tools(fresh, preserve_prefix=True)
        assert json.dumps(fresh.tools) == before
        with bind_agent_config(fresh):
            from tools.tool_search import resolve_underlying_call
            assert resolve_underlying_call({"calls": [{"name": ADMIN, "arguments": {}}]})[0] == ADMIN
            assert resolve_underlying_call({"calls": [{"name": IMAGE, "arguments": {}}]})[2]


def test_legacy_pin_never_adopts_new_eager_names(configured, registered):
    from tools.mcp_tool_agent import restore_agent_tool_prefix, tool_pin_version
    legacy = _agent(configured({"eager": []}))
    saved = {"version": tool_pin_version(), "tools": legacy.tools}
    new = _agent(configured({"eager": [IMAGE]}))
    restore_agent_tool_prefix(new, saved)
    assert agent_config(new).eager_tools == set()
    assert IMAGE not in new.valid_tool_names and ADMIN not in new.valid_tool_names
    assert new.tools == saved["tools"]


def test_missing_pin_does_not_clear_a_new_sessions_eager_policy(configured, registered):
    from tools.mcp_tool_agent import restore_agent_tool_prefix
    agent = _agent(configured({"eager": [IMAGE]}))
    for missing in (None, [], {}, {"version": "legacy", "tools": []}):
        assert not restore_agent_tool_prefix(agent, missing)
        assert agent_config(agent).eager_tools == {IMAGE}


def test_frozen_policy_is_context_local_and_cache_isolated(configured, registered):
    import model_tools
    first = _agent(configured({"eager": [IMAGE]}))
    second = _agent(configured({"eager": [ADMIN]}))
    for agent, direct, deferred in ((first, IMAGE, ADMIN), (second, ADMIN, IMAGE), (first, IMAGE, ADMIN)):
        with bind_agent_config(agent):
            definitions = model_tools.get_tool_definitions(enabled_toolsets=[TOOLSET], quiet_mode=True)
            assert direct in _names(definitions) and deferred not in _names(definitions)
    assert load_config_readonly().eager_tools == {ADMIN}


def test_executor_paths_keep_the_session_policy_after_config_edit(configured, registered):
    from agent.agent_runtime_helpers import invoke_tool
    from agent.tool_executor import _resolve_sequential_dispatch, _ToolCallRef, _unwrap_tool_search_call
    agent = _agent(configured({"eager": [IMAGE]}))
    agent._memory_manager, agent._context_engine_tool_names, agent.quiet_mode = None, set(), False
    configured({"eager": [ADMIN]})
    name, _, blocked = _unwrap_tool_search_call(
        agent, "tool_call", {"calls": [{"name": ADMIN, "arguments": {}}]})
    assert name == ADMIN and blocked is None
    args = {"queries": ["probe"]}
    concurrent = invoke_tool(agent, "tool_search", args, "probe-task", pre_tool_block_checked=True)
    sequential = _resolve_sequential_dispatch(
        agent, _ToolCallRef("tool_search", args, "probe-task", "probe-call", []), [])
    for result in (concurrent, sequential.execute(args)):
        found = json.loads(result)["tools"]
        assert ADMIN in found and IMAGE not in found


def test_profile_configs_cannot_share_eager_policy(configured, registered, tmp_path, monkeypatch):
    from hermes_cli.config import atomic_config_write
    first = _agent(configured({"eager": [IMAGE]}))
    other = tmp_path / "other-profile"
    other.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(other))
    atomic_config_write(other / "config.yaml", {"tools": {"tool_search": {"eager": [ADMIN]}}})
    assert load_config_readonly().eager_tools == {ADMIN}
    with bind_agent_config(first):
        assert load_config_readonly().eager_tools == {IMAGE}
    assert load_config_readonly().eager_tools == {ADMIN}


@pytest.mark.parametrize("separate_profile", [False, True])
def test_nested_new_agent_reads_its_own_profile_policy(
    configured, registered, tmp_path, monkeypatch, separate_profile,
):
    import model_tools
    from hermes_cli.config import atomic_config_write
    from tools.tool_search_policy import build_agent_tool_defs

    parent = _agent(configured({"eager": [IMAGE]}))
    if separate_profile:
        child_home = tmp_path / "child-profile"
        child_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(child_home))
        atomic_config_write(child_home / "config.yaml", {"tools": {"tool_search": {"eager": [ADMIN]}}})
    else:
        configured({"eager": [ADMIN]})
    child = SimpleNamespace()
    with bind_agent_config(parent):
        definitions = build_agent_tool_defs(child, model_tools.get_tool_definitions, [TOOLSET], [])
        assert ADMIN in _names(definitions) and IMAGE not in _names(definitions)
        assert agent_config(child).eager_tools == {ADMIN}
        # Initializing the child must leave its parent's frozen dispatch scope intact.
        assert load_config_readonly().eager_tools == {IMAGE}
    assert load_config_readonly().eager_tools == {ADMIN}


def test_tool_history_projection_preserves_eager_definitions(configured, registered, tmp_path):
    import copy
    import model_tools
    from agent.tool_history_policy import bind_tool_history_for_turn, project_tool_history_for_request
    from hermes_cli.config import atomic_config_write
    from tools.tool_search_policy import build_agent_tool_defs, dispatch_agent_tool

    config = configured({"eager": [IMAGE]})
    atomic_config_write(tmp_path / "config.yaml", {
        "tools": {"tool_search": {"eager": [IMAGE]}}, "agent": {"tool_history_echo": False},
    })
    agent = _agent(config)
    agent.api_mode, agent.provider, agent.base_url = "chat_completions", "openai", "https://example.invalid/v1"
    agent.client = SimpleNamespace()
    messages = [
        {"role": "user", "content": "Create a probe image."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "probe-call", "type": "function",
         "function": {"name": IMAGE, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "probe-call", "content": "Probe image created."},
        {"role": "assistant", "content": "The image is ready."},
        {"role": "user", "content": "Change its color."},
    ]
    bind_tool_history_for_turn(agent, messages)
    canonical = copy.deepcopy(messages)
    schemas = json.dumps(agent.tools)
    projected = project_tool_history_for_request(agent, messages).messages
    assert all(row["role"] != "tool" and not row.get("tool_calls") for row in projected)
    assert any(row.get("content") == "The image is ready." for row in projected)
    assert messages == canonical
    rebuilt = build_agent_tool_defs(agent, model_tools.get_tool_definitions, [TOOLSET], [])
    assert json.dumps(rebuilt) == schemas == json.dumps(agent.tools)
    assert IMAGE in agent.valid_tool_names
    result = dispatch_agent_tool(agent, IMAGE, {}, "probe-task", enabled_toolsets=[TOOLSET])
    assert json.loads(result)["success"]
