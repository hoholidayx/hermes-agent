"""Session-frozen Tool Search routing, scoped across registry builds and dispatch.

The live registry still owns availability and authorization. This policy only decides
whether a permitted tool is offered directly or reached through the bridge.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace


_ACTIVE_CONFIG = ContextVar("tool_search_session_config", default=None)


def active_config():
    return _ACTIVE_CONFIG.get()


def agent_config(agent):
    from tools.tool_search import ToolSearchConfig, load_config_readonly
    config = getattr(agent, "_tool_search_config", None)
    if not isinstance(config, ToolSearchConfig):
        # A delegated/new agent can be constructed inside its parent's dispatch.
        # Read this agent's profile, not the parent's temporarily bound routing.
        token = _ACTIVE_CONFIG.set(None)
        try:
            config = load_config_readonly()
        finally:
            _ACTIVE_CONFIG.reset(token)
        agent._tool_search_config = config
    return config


@contextmanager
def bind_agent_config(agent):
    token = _ACTIVE_CONFIG.set(agent_config(agent))
    try:
        yield
    finally:
        _ACTIVE_CONFIG.reset(token)


def build_agent_tool_defs(agent, builder, enabled, disabled, *, quiet_mode=True):
    """Build through the regular registry filters, under this session's frozen routing."""
    with bind_agent_config(agent):
        return builder(enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=quiet_mode) or []


def dispatch_agent_tool(agent, name, args, task_id, **kwargs):
    """Registry dispatch shared by executor threads; bind before any bridge/catalog read."""
    import model_tools
    with bind_agent_config(agent):
        return model_tools.handle_function_call(name, args, task_id, **kwargs)


def restore_agent_config(agent, saved):
    """Restore the pin's routing before rebuilding. Legacy pins never opted into eager."""
    from tools.tool_search import ToolSearchConfig
    if not (saved.get("tools") if isinstance(saved, dict) else saved):
        return False
    previous = getattr(agent, "_tool_search_config", None)
    current = agent_config(agent)
    raw = saved.get("tool_search") if isinstance(saved, dict) else None
    config = (ToolSearchConfig.from_raw(raw) if isinstance(raw, dict)
              else replace(current, eager_tools=frozenset()))
    agent._tool_search_config = config
    return isinstance(previous, ToolSearchConfig) and previous != config
