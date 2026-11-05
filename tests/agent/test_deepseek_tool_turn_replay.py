"""DeepSeek replay filters previous user turns while retaining current reasoning."""

from copy import deepcopy

import pytest

from agent.message_sanitization import REASONING_CARRIERS, project_reasoning_for_request
from run_agent import AIAgent


@pytest.mark.parametrize("provider,model,base_url", [
    ("DeepSeek", "alias", "https://example.com/v1"),
    ("sudorelay", "deepseek-v4-1-flash", "https://example.com/v1"),
    ("custom", "alias", "https://api.deepseek.com/v1"),
    ("openrouter", "deepseek/deepseek-v4", "https://openrouter.ai/api/v1"),
    ("custom", "deepseek-v4", "http://127.0.0.1:8080/v1"),
])
@pytest.mark.parametrize("tool_calls", [None, []])
@pytest.mark.parametrize("reasoning_fields", [
    {"reasoning_content": "thought", "reasoning": "thought",
     "reasoning_details": [{"type": "reasoning.text", "text": "thought"}]},
    {"reasoning": "alias-only thought"},
    {"reasoning_details": [{"type": "reasoning.text", "text": "details-only thought"}]},
    {"reasoning_content": ""},
    {},
])
def test_deepseek_plain_turn_strips_every_carrier(provider, model, base_url, tool_calls, reasoning_fields):
    agent = object.__new__(AIAgent)
    agent.provider, agent.model, agent.base_url = provider, model, base_url
    agent.api_mode = "chat_completions"
    agent._reasoning_echo_flag = False
    source = {"role": "assistant", "content": "answer", **deepcopy(reasoning_fields)}
    if tool_calls is not None:
        source["tool_calls"] = tool_calls
    original = deepcopy(source)
    api = deepcopy(source)

    messages = project_reasoning_for_request([
        {"role": "user", "content": "previous"}, source,
        {"role": "user", "content": "next"},
    ], agent._reasoning_replay_route())
    api = messages[1]

    assert not set(REASONING_CARRIERS).intersection(api)
    assert api["content"] == "answer"
    assert source == original
    # The retry/fallback seam must not put stripped reasoning (or a pad) back.
    assert agent._reapply_reasoning_echo_for_provider(messages) == 0
    assert not set(REASONING_CARRIERS).intersection(api)


@pytest.mark.parametrize("provider,model,base_url,api_mode,expected_keys", [
    ("openrouter", "deepseek/deepseek-v4", "https://openrouter.ai/api/v1", "chat_completions",
     set(REASONING_CARRIERS)),
    ("sudorelay", "deepseek-v4-1-flash", "https://example.com/v1", "chat_completions",
     {"reasoning_content"}),
    ("openrouter", "qwen/qwen3", "https://openrouter.ai/api/v1", "chat_completions",
     set(REASONING_CARRIERS)),
    ("kimi-coding", "kimi-k3", "https://api.kimi.com/v1", "chat_completions", {"reasoning_content"}),
    ("xiaomi", "mimo", "https://api.xiaomimimo.com/v1", "chat_completions", {"reasoning_content"}),
    ("custom", "qwen3", "http://127.0.0.1:8080/v1", "chat_completions", {"reasoning_content", "reasoning"}),
    ("custom", "deepseek-v4", "https://api.deepseek.com/anthropic", "anthropic_messages",
     {"reasoning_content", "reasoning_details"}),
])
@pytest.mark.parametrize("has_tools", [False, True])
def test_tool_reasoning_and_other_routes_keep_existing_replay(
    provider, model, base_url, api_mode, expected_keys, has_tools,
):
    agent = object.__new__(AIAgent)
    agent.provider, agent.model, agent.base_url = provider, model, base_url
    agent.api_mode = api_mode
    source = {"role": "assistant", "content": "answer", "reasoning_content": "thought", "reasoning": "thought",
              "reasoning_details": [{"type": "reasoning.text", "text": "thought"}]}
    if has_tools:
        source["tool_calls"] = [{"id": "c", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
    original = deepcopy(source)
    api = deepcopy(source)

    agent._copy_reasoning_content_for_api(source, api)

    assert set(REASONING_CARRIERS).intersection(api) == expected_keys
    if "reasoning_content" in expected_keys:
        assert api["reasoning_content"] == "thought"
    if "reasoning_details" in expected_keys:
        assert api["reasoning_details"] == source["reasoning_details"]
    assert source == original
    assert agent._reapply_reasoning_echo_for_provider([api]) == 0


def test_fallback_to_deepseek_strips_prebuilt_plain_turn_but_keeps_tool_turn():
    agent = object.__new__(AIAgent)
    agent.provider, agent.model, agent.base_url = "openrouter", "qwen/qwen3", "https://openrouter.ai/api/v1"
    agent.api_mode = "chat_completions"
    plain = {"role": "assistant", "content": "answer", "reasoning_content": "thought", "reasoning": "thought",
             "reasoning_details": [{"type": "reasoning.text", "text": "thought"}]}
    tool = {**deepcopy(plain), "tool_calls": [{"id": "c", "function": {"name": "t", "arguments": "{}"}}]}
    agent._reasoning_echo_flag = True
    messages = [{"role": "user", "content": "previous"}, plain, {"role": "user", "content": "next"}, tool]
    assert agent._reapply_reasoning_echo_for_provider(messages) == 0

    agent.model = "deepseek/deepseek-v4"
    agent._reasoning_echo_flag = False

    assert agent._reapply_reasoning_echo_for_provider(messages) == 1
    assert not set(REASONING_CARRIERS).intersection(messages[1])
    assert tool["reasoning_content"] == "thought"
    assert tool["reasoning"] == "thought"
    assert tool["reasoning_details"] == [{"type": "reasoning.text", "text": "thought"}]
    assert agent._reapply_reasoning_echo_for_provider(messages) == 0
