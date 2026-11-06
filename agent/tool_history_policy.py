"""Opt-in, primary-chat tool-history projection; durable transcripts stay intact."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolHistoryProjection:
    messages: list[dict]
    source_indices: list[int]


def read_tool_history_echo(agent: Any) -> bool:
    """Only a YAML boolean false opts out; reads honor the owning session/profile."""
    from hermes_cli.config import load_config_readonly
    from gateway.session_policy import active_policy

    policy = active_policy()
    config = policy.config(secrets=False) if policy is not None and not policy.follows_profile else load_config_readonly()
    section = config.get("agent")
    value = section.get("tool_history_echo", True) if isinstance(section, dict) else True
    if not isinstance(value, bool):
        logger.warning("agent.tool_history_echo must be a boolean; retaining tool history")
    return value is not False


def is_primary_tool_history_session(agent: Any) -> bool:
    from agent.delegation_context import is_delegated_child_process_context, owned_kanban_task
    from agent.session_source import session_source_for

    platform = getattr(agent, "platform", None)
    if (getattr(agent, "side_agent", False) or getattr(agent, "_parent_session_id", None)
            or getattr(agent, "_delegate_depth", 0) or platform in {"subagent", "cron", "kanban"}):
        return False
    return not (is_delegated_child_process_context() or owned_kanban_task()
                or session_source_for(platform) in {"cron", "kanban"})


def tool_history_transport_supported(agent: Any) -> bool:
    """Some clients expose a completions facade over a native/server-owned protocol."""
    from agent.gemini_native_adapter import is_native_gemini_base_url
    from hermes_cli.runtime_provider_backends import _is_external_process_provider

    url = str(getattr(agent, "base_url", "") or "")
    provider = str(getattr(agent, "provider", "") or "")
    client = getattr(agent, "client", None)
    return (
        getattr(agent, "api_mode", "") == "chat_completions"
        and provider != "moa" and not _is_external_process_provider(provider)
        and not url.lower().startswith(("acp://", "acp+tcp://"))
        and not is_native_gemini_base_url(url)
        and getattr(client, "HERMES_SKIP_TRANSPORT_WRAP", None) is not True
        and getattr(client, "SUPPORTS_HERMES_TOOL_CALLS", None) is not False
    )


def tool_history_filter_requested(agent: Any) -> bool:
    if not hasattr(agent, "_tool_history_echo_flag"):
        agent._tool_history_echo_flag = read_tool_history_echo(agent)
    primary = getattr(agent, "_tool_history_primary", None)
    if primary is None:
        primary = is_primary_tool_history_session(agent)
    return getattr(agent, "_tool_history_echo_flag", True) is False and bool(primary)


def effective_tool_history_filter(agent: Any) -> bool:
    return tool_history_filter_requested(agent) and tool_history_transport_supported(agent)


def _latest_user_start(messages: list) -> int:
    from agent.context_compressor import is_user_originated_turn
    from agent.prompt_builder import STEER_DISPLAY_KIND

    return next((i for i in range(len(messages) - 1, -1, -1)
                 if is_user_originated_turn(messages[i])
                 and messages[i].get("display_kind") != STEER_DISPLAY_KIND), -1)


def _scope_start(agent: Any, messages: list) -> int:
    from agent.message_metadata import ABSORBED_MESSAGE_UIDS, MESSAGE_UID

    boundary = getattr(agent, "_tool_history_boundary", None)
    if boundary is None:
        return _latest_user_start(messages)
    # Compaction/selection may replace the anchor. An unproven boundary cannot retire tools.
    return next((i for i, msg in enumerate(messages)
                 if msg.get(MESSAGE_UID) == boundary or boundary in (msg.get(ABSORBED_MESSAGE_UIDS) or ())), -1)


def _boundary_key(messages: list, start: int) -> str:
    from agent.usage_anchor import message_fingerprint

    raw = json.dumps([start, message_fingerprint(messages[start]) if start >= 0 else None])
    return hashlib.sha256(raw.encode()).hexdigest()


def tool_history_boundary_key(messages: list) -> str:
    return _boundary_key(messages, _latest_user_start(messages))


def tool_history_scope_key(agent: Any, messages: list) -> dict | None:
    if not effective_tool_history_filter(agent):
        return None
    return {"boundary": _boundary_key(messages, _scope_start(agent, messages))}


def bind_tool_history_for_turn(agent: Any, messages: list) -> None:
    """Freeze policy and the admitted human boundary, never advancing it for /steer."""
    from agent.message_metadata import stamp_message_uid

    start = _latest_user_start(messages)
    # Internal continuation turns keep the preceding human task and its frozen setting.
    if start == len(messages) - 1 or not hasattr(agent, "_tool_history_primary"):
        agent._tool_history_echo_flag = read_tool_history_echo(agent)
        agent._tool_history_primary = is_primary_tool_history_session(agent)
    agent._tool_history_boundary = stamp_message_uid(messages[start]) if start >= 0 else None


def warn_unsupported_tool_history(agent: Any) -> None:
    if not tool_history_filter_requested(agent):
        return
    if tool_history_transport_supported(agent):
        return
    route = (getattr(agent, "api_mode", ""), getattr(agent, "provider", ""), getattr(agent, "base_url", ""))
    warned = getattr(agent, "_tool_history_unsupported_routes", set())
    if route in warned:
        return
    agent._tool_history_unsupported_routes = warned | {route}
    text = "agent.tool_history_echo=false supports Chat Completions only; this route retains tool history."
    logger.warning(text)
    emit = getattr(agent, "_emit_warning", None)
    if callable(emit):
        emit(text)


def project_tool_history_for_request(agent: Any, messages: list) -> ToolHistoryProjection:
    """Strip old tool records on independent copies, preserving assistant visible content."""
    projected, indices = [], []
    start = _scope_start(agent, messages) if effective_tool_history_filter(agent) else -1
    for index, source in enumerate(messages):
        if index < start and source.get("role") == "tool":
            continue
        target = deepcopy(source)
        if index < start and source.get("role") == "assistant":
            api_content = target.pop("api_content", None)
            if isinstance(api_content, str) and api_content:
                target["content"] = api_content
        if index < start and source.get("role") == "assistant" and source.get("tool_calls"):
            target.pop("tool_calls", None)
            # Native records can contain an entire assistant turn, including the removed calls.
            for key in ("extra_content", "anthropic_content_blocks", "_anthropic_content_blocks",
                        "bedrock_content_blocks", "codex_message_items", "codex_reasoning_items"):
                target.pop(key, None)
            details = target.get("reasoning_details")
            if isinstance(details, list):
                target["reasoning_details"] = [d for d in details if not isinstance(d, dict)
                                               or not str(d.get("type", "")).endswith(".native_assistant")]
            content = target.get("content")
            if not content or (isinstance(content, str) and not content.strip()):
                continue
        projected.append(target)
        indices.append(index)
    return ToolHistoryProjection(projected, indices)


def _merge_history_reasoning(previous: dict, message: dict) -> None:
    for key in ("reasoning", "reasoning_content"):
        parts = [m.get(key) for m in (previous, message) if isinstance(m.get(key), str) and m[key]]
        if parts:
            previous[key] = "\n\n".join(parts)
    if message.get("reasoning_details"):
        previous["reasoning_details"] = [*(previous.get("reasoning_details") or []), *message["reasoning_details"]]


def merge_tool_history_assistants(agent: Any, messages: list) -> list:
    """Join the assistant text exposed by deleting intervening history-only tool rows."""
    if not effective_tool_history_filter(agent):
        return messages
    from agent.agent_runtime_helpers import _merge_user_content, _UNMERGEABLE

    merged = []
    start = _scope_start(agent, messages)
    for index, message in enumerate(messages):
        previous = merged[-1] if merged else None
        if (index >= start or previous is None or previous.get("role") != "assistant" or message.get("role") != "assistant"
                or previous.get("tool_calls") or message.get("tool_calls")):
            merged.append(message)
            continue
        content = _merge_user_content(previous.get("content", ""), message.get("content", ""))
        if content is _UNMERGEABLE:
            merged.append(message)
            continue
        previous["content"] = content
        previous.pop("api_content", None)
        _merge_history_reasoning(previous, message)
    return merged
