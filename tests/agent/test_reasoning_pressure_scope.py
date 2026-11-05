"""Compression pressure follows the sent user-turn reasoning, including all tool iterations."""

from copy import deepcopy
from types import SimpleNamespace

from agent.context_compressor_tail import tail_row_pricer
from agent.turn_context import _preflight_request_tokens
from agent.usage_anchor import capture_usage_anchor


def _route():
    return SimpleNamespace(api_mode='chat_completions', provider='deepseek', model='deepseek-flash',
                           base_url='https://api.deepseek.com/v1', _reasoning_echo_flag=False,
                           tools=[], context_compressor=SimpleNamespace(
                               last_real_prompt_tokens=0, last_compression_rough_tokens=0))


def _tool(call_id, chars):
    return {'role': 'assistant', 'content': '', 'reasoning_content': 'x' * chars,
            'tool_calls': [{'id': call_id, 'type': 'function',
                            'function': {'name': 'lookup', 'arguments': '{}'}}]}


def _history():
    return [{'role': 'user', 'content': 'previous'}, _tool('old', 200_000),
            {'role': 'tool', 'tool_call_id': 'old', 'content': 'old result'},
            {'role': 'assistant', 'content': 'done'}, {'role': 'user', 'content': 'current'},
            _tool('first', 40_000), {'role': 'tool', 'tool_call_id': 'first', 'content': 'one'},
            _tool('second', 60_000), {'role': 'tool', 'tool_call_id': 'second', 'content': 'two'}]


def test_preflight_and_tail_count_all_current_tool_reasoning_without_old_reasoning():
    agent = _route()
    history = _history()
    original = deepcopy(history)

    estimate = _preflight_request_tokens(agent, history, '')
    price = tail_row_pricer(agent, history)

    # 100K current chars is roughly 25K tokens. Counting only the newest iteration
    # would give 15K; including the previous user turn would give 75K.
    assert 25_000 <= estimate < 26_000
    assert price(1) < 100
    assert price(5) >= 10_000 and price(7) >= 15_000
    assert history == original


def test_next_user_retires_old_usage_floor_and_all_previous_tool_reasoning():
    agent = _route()
    history = _history()
    agent._usage_anchor = capture_usage_anchor(75_000, 10_000, history)
    agent.context_compressor.last_real_prompt_tokens = 75_000
    agent.context_compressor.last_compression_rough_tokens = 75_000
    history += [{'role': 'assistant', 'content': 'done'}, {'role': 'user', 'content': 'hello'}]

    estimate = _preflight_request_tokens(agent, history, '')

    assert estimate < 500
    assert agent._usage_anchor is None
    assert agent.context_compressor.last_real_prompt_tokens == 0
