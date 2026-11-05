"""Registered non-OpenAI provider routes obey one explicit cross-user replay setting.

This enumerates the real registry, so newly bundled providers join the behavioural
matrix automatically. Native protocols are classified separately instead of being
silently claimed as covered by the Chat Completions setting.
"""

from copy import deepcopy

import pytest

from agent.message_sanitization import (
    REASONING_CARRIERS,
    project_reasoning_for_request,
    reasoning_replay_route,
)
from agent.route_binding import reasoning_echo_for_fallback
from providers import list_providers


_PROFILES = tuple(p for p in list_providers() if p.name not in {'openai', 'openai-api', 'openai-codex'})
_CHAT_PROFILES = tuple(p for p in _PROFILES if p.api_mode == 'chat_completions')
_NATIVE_PROFILES = tuple(p for p in _PROFILES if p.api_mode != 'chat_completions')


def _route(profile, enabled=False):
    model = next(iter(profile.fallback_models), 'demo-model')
    return reasoning_replay_route(profile.api_mode, profile.name, model, profile.base_url,
                                  echo_opt_in=enabled)


def _assistant(label, call_id):
    return {'role': 'assistant', 'content': label,
            'reasoning_content': label + ' thought', 'reasoning': label + ' thought',
            'reasoning_details': [{'type': 'reasoning.text', 'text': label + ' thought'}],
            'tool_calls': [{'id': call_id, 'type': 'function',
                            'function': {'name': 'lookup', 'arguments': '{}'}}]}


def _history():
    return [{'role': 'user', 'content': 'old question'}, _assistant('old', 'old-call'),
            {'role': 'tool', 'tool_call_id': 'old-call', 'content': 'old result'},
            {'role': 'assistant', 'content': 'old final', 'reasoning_content': 'old final thought'},
            {'role': 'user', 'content': 'new question'}, _assistant('current', 'current-call'),
            {'role': 'tool', 'tool_call_id': 'current-call', 'content': 'new result'}]


def test_provider_matrix_is_populated_by_real_discovery():
    assert _CHAT_PROFILES
    assert _NATIVE_PROFILES


@pytest.mark.parametrize('profile', _CHAT_PROFILES, ids=lambda p: p.name)
def test_every_chat_provider_defaults_to_current_user_turn_without_losing_tools(profile):
    history = _history()
    original = deepcopy(history)
    route = _route(profile)

    projected = project_reasoning_for_request(history, route)

    assert route.replay_history is False
    assert not set(REASONING_CARRIERS).intersection(projected[1])
    assert not set(REASONING_CARRIERS).intersection(projected[3])
    assert set(REASONING_CARRIERS).intersection(projected[5]) == set(route.carriers)
    if 'reasoning_content' in route.carriers:
        assert projected[5]['reasoning_content'] == 'current thought'
    assert projected[1]['tool_calls'] == history[1]['tool_calls']
    assert projected[2] == history[2]
    assert history == original


@pytest.mark.parametrize('profile', _CHAT_PROFILES, ids=lambda p: p.name)
def test_every_chat_provider_requires_explicit_opt_in_for_history(profile):
    history = _history()
    route = _route(profile, enabled=True)

    projected = project_reasoning_for_request(history, route)

    assert route.replay_history is True
    assert set(REASONING_CARRIERS).intersection(projected[1]) == set(route.carriers)
    if 'reasoning_content' in route.carriers:
        assert projected[1]['reasoning_content'] == 'old thought'
    assert history[1]['reasoning_content'] == 'old thought'


@pytest.mark.parametrize('profile', _CHAT_PROFILES, ids=lambda p: p.name)
def test_provider_required_pads_never_reenable_completed_turn_reasoning(profile):
    history = _history()
    for message in history:
        for key in REASONING_CARRIERS:
            message.pop(key, None)
    route = _route(profile)

    projected = project_reasoning_for_request(history, route)

    assert not set(REASONING_CARRIERS).intersection(projected[1])
    assert not set(REASONING_CARRIERS).intersection(projected[3])
    if route.pad:
        assert projected[5]['reasoning_content'] == ' '
    else:
        assert 'reasoning_content' not in projected[5]


@pytest.mark.parametrize('profile', _CHAT_PROFILES, ids=lambda p: p.name)
def test_fallback_setting_inherits_primary_and_explicit_false_wins_for_every_chat_provider(profile):
    from types import SimpleNamespace

    agent = SimpleNamespace(api_mode=profile.api_mode, _reasoning_echo_flag=False,
                            _primary_runtime={'reasoning_echo_flag': True})

    assert reasoning_echo_for_fallback(agent, {}) is True
    assert reasoning_echo_for_fallback(agent, {'reasoning_echo': False}) is False
    agent._primary_runtime['reasoning_echo_flag'] = False
    assert reasoning_echo_for_fallback(agent, {}) is False
    assert reasoning_echo_for_fallback(agent, {'reasoning_echo': True}) is True


@pytest.mark.parametrize('profile', _NATIVE_PROFILES, ids=lambda p: p.name)
def test_native_provider_contracts_remain_outside_chat_replay_scope(profile):
    disabled, enabled = _route(profile), _route(profile, enabled=True)

    assert disabled.carriers is None
    assert enabled.carriers is None
    assert disabled.replay_history is False and enabled.replay_history is False


@pytest.mark.parametrize('provider', ('mistral', 'groq', 'cerebras', 'sambanova'))
@pytest.mark.parametrize('enabled', (False, True))
def test_strict_compatibility_routes_never_receive_reasoning_even_for_deepseek_models(provider, enabled):
    # These route IDs remain supported by the central wire policy, even when a
    # checkout has no bundled provider profile for the service.
    route = reasoning_replay_route('chat_completions', provider, 'deepseek-proxy-model',
                                  'https://example.com/v1', echo_opt_in=enabled)

    projected = project_reasoning_for_request(_history(), route)

    assert not route.carriers and not route.pad
    assert not any(set(REASONING_CARRIERS).intersection(message) for message in projected)


@pytest.mark.parametrize('name,model', (
    ('sudorelay', 'deepseek-v4-1-flash'),
    ('xkrelay', 'deepseek-v4.1-flash-1017'),
))
def test_registered_deepseek_relays_share_the_disabled_default(tmp_path, monkeypatch, name, model):
    import providers
    from providers.base import ProviderProfile
    from run_agent import AIAgent

    # Seed the same route identity without importing the operator's user plugin
    # code, endpoint settings or credentials into the portable test process.
    monkeypatch.setattr(providers, '_REGISTRY', dict(providers._REGISTRY))
    monkeypatch.setattr(providers, '_SOURCES', dict(providers._SOURCES))
    monkeypatch.setattr(providers, '_PROVIDER_LIST_CACHE', None)
    profile = ProviderProfile(name=name, base_url='https://relay.example/v1', fallback_models=(model,))
    providers.register_provider(profile)
    assert providers.get_provider_profile(name) is profile
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(
        f'model:\n  provider: {name}\n  default: {model}\n', encoding='utf-8')
    agent = object.__new__(AIAgent)
    agent.api_mode, agent.provider, agent.model, agent.base_url = (
        'chat_completions', name, model, profile.base_url)
    agent._reasoning_echo_flag = agent._read_reasoning_echo_from_config()

    route = agent._reasoning_replay_route()
    projected = project_reasoning_for_request(_history(), route)

    assert agent._reasoning_echo_flag is False and route.replay_history is False
    assert route.pad is True  # Current tool-loop continuity is separate from historical echo.
    assert not set(REASONING_CARRIERS).intersection(projected[1])
    assert projected[5]['reasoning_content'] == 'current thought'
