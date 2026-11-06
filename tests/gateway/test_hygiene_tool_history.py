"""Gateway hygiene prices the next primary tool-history projection before admission."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.run_turn_hygiene import GatewayTurnHygieneMixin


def _history():
    return [
        {"role": "user", "content": "previous"},
        {"role": "assistant", "content": "I will inspect it.", "tool_calls": [{
            "id": "old", "type": "function", "function": {"name": "lookup", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "old", "content": "x" * 200_000},
        {"role": "assistant", "content": "done"},
    ]


def _settings(*, echo=False, api_mode="chat_completions", hard_msg_limit=5000):
    return SimpleNamespace(
        model="test-model", provider="custom", base_url="http://127.0.0.1:8123/v1", api_key="",
        api_mode=api_mode, data={"model": {"reasoning_echo": True}, "agent": {"tool_history_echo": echo}},
        config_context_length=10_000, threshold_pct=.85, hard_msg_limit=hard_msg_limit,
    )


def _runner(monkeypatch):
    monkeypatch.setattr("agent.model_metadata.get_model_context_length_async", AsyncMock(return_value=10_000))
    runner = GatewayTurnHygieneMixin()
    runner._session_has_compression_in_flight = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._session_db._db.get_compression_failure_cooldown.return_value = None
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("last_prompt", [0, 100_000])
async def test_next_user_drops_historical_tools_and_ignores_retired_usage(monkeypatch, last_prompt):
    history = _history()
    original = deepcopy(history)
    runner = _runner(monkeypatch)

    plan = await runner._hmwa_hygiene_plan(
        _settings(), history, SimpleNamespace(session_id="tools", last_prompt_tokens=last_prompt), "key",
        incoming_message={"role": "user", "content": "next" + "n" * 100_000},
    )

    assert plan.approx_tokens < 100  # Incoming payload is not part of the history budget.
    assert not plan.needs_compress
    assert history == original
    runner._session_db._db.get_session_model_config_value.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [_settings(echo=True), _settings(api_mode="anthropic_messages")])
async def test_disabled_or_unsupported_filter_preserves_actual_usage(monkeypatch, settings):
    runner = _runner(monkeypatch)

    plan = await runner._hmwa_hygiene_plan(
        settings, _history(), SimpleNamespace(session_id="tools", last_prompt_tokens=100_000), "key",
        incoming_message={"role": "user", "content": "next"},
    )

    assert plan.approx_tokens == 100_000
    assert plan.needs_compress


@pytest.mark.asyncio
@pytest.mark.parametrize("display_kind", ["internal_notification", "steer"])
async def test_synthetic_and_steer_inputs_keep_active_tool_chain(monkeypatch, display_kind):
    runner = _runner(monkeypatch)

    plan = await runner._hmwa_hygiene_plan(
        _settings(), _history()[:-1], SimpleNamespace(session_id="tools", last_prompt_tokens=0), "key",
        incoming_message={"role": "user", "content": "continue", "display_kind": display_kind},
    )

    assert plan.approx_tokens >= 50_000
    assert plan.needs_compress


@pytest.mark.asyncio
async def test_raw_hard_message_limit_still_protects_large_transcripts(monkeypatch):
    runner = _runner(monkeypatch)

    plan = await runner._hmwa_hygiene_plan(
        _settings(hard_msg_limit=4), _history(), SimpleNamespace(session_id="tools", last_prompt_tokens=0), "key",
        incoming_message={"role": "user", "content": "next"},
    )

    assert plan.approx_tokens < 100
    assert plan.needs_compress


def _policy_runner(tmp_path, monkeypatch, *, echo, following=False, worker=None, provider="custom"):
    from gateway import run
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_local import LocalSessionAdapter
    from gateway.session_policy import build_policy
    from hermes_cli.config import atomic_config_write

    config = {"model": {"default": "test-model", "provider": provider, "reasoning_echo": True,
                        "base_url": "http://127.0.0.1:8123/v1", "context_length": 10_000},
              "agent": {"tool_history_echo": echo}}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(run, "_hermes_home", tmp_path)
    atomic_config_write(tmp_path / "config.yaml", config)
    policy = build_policy({"source": "cli", "cwd": str(tmp_path), "follow_profile_config": following}, config)
    if worker is not None:
        policy = replace(policy, source=worker)
    source = SessionSource(platform=Platform.LOCAL, chat_id="policy-tools")
    authority = SimpleNamespace(db=SimpleNamespace(db_path=tmp_path / "state.db"),
                                sessions={"policy-tools": SimpleNamespace(source=source)})
    adapter = LocalSessionAdapter(authority)
    adapter.register_source(source)
    adapter.policies[source.chat_id] = policy
    runner = _runner(monkeypatch)
    runner._HygieneSettings = run.GatewayRunner._HygieneSettings
    runner.session_authority = authority
    runner._adapter_for_source = lambda _source: adapter
    # Model resolution is independent of replay policy and cannot contact a provider here.
    runner._resolve_session_agent_runtime = lambda **_kwargs: ("test-model", {
        "provider": provider, "base_url": "http://127.0.0.1:8123/v1", "api_mode": "chat_completions",
    })
    return runner, source, config


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("following", [False, True], ids=["frozen", "following"])
async def test_hygiene_uses_the_sessions_tool_policy_after_profile_changes(tmp_path, monkeypatch, echo, following):
    from hermes_cli.config import atomic_config_write

    runner, source, config = _policy_runner(tmp_path, monkeypatch, echo=echo, following=following)
    atomic_config_write(tmp_path / "config.yaml", {**config, "agent": {"tool_history_echo": not echo}})
    settings = await runner._hmwa_hygiene_settings(source, "key")

    plan = await runner._hmwa_hygiene_plan(
        settings, _history(), SimpleNamespace(session_id="tools", last_prompt_tokens=100_000), "key",
        incoming_message={"role": "user", "content": "next"},
    )

    expected_echo = not echo if following else echo
    assert plan.needs_compress is expected_echo
    assert (plan.approx_tokens >= 50_000) is expected_echo
    assert settings.data["agent"]["tool_history_echo"] is not echo


@pytest.mark.asyncio
@pytest.mark.parametrize("worker", ["cron", "kanban"])
async def test_hygiene_keeps_tool_context_for_worker_policy_sources(tmp_path, monkeypatch, worker):
    runner, source, _config = _policy_runner(tmp_path, monkeypatch, echo=False, worker=worker)
    settings = await runner._hmwa_hygiene_settings(source, "key")

    plan = await runner._hmwa_hygiene_plan(
        settings, _history(), SimpleNamespace(session_id="tools", last_prompt_tokens=100_000), "key",
        incoming_message={"role": "user", "content": "next"},
    )

    assert plan.needs_compress
    assert plan.approx_tokens == 100_000


@pytest.mark.asyncio
@pytest.mark.parametrize("supplied_client", [False, True], ids=["standard-client", "profile-client"])
async def test_hygiene_prices_unknown_profile_clients_conservatively_without_constructing_them(
    tmp_path, monkeypatch, supplied_client,
):
    import providers
    from providers.base import ProviderProfile

    calls = []

    class NativeProfile(ProviderProfile):
        def create_client(self, **_kwargs):
            calls.append(True)
            return SimpleNamespace(HERMES_SKIP_TRANSPORT_WRAP=True)

    profile_type = NativeProfile if supplied_client else ProviderProfile
    profile = profile_type(name="hygiene-fixture", base_url="http://127.0.0.1:8123/v1")
    providers.list_providers()
    providers.register_provider(profile)
    try:
        runner, source, _config = _policy_runner(
            tmp_path, monkeypatch, echo=False, provider=profile.name,
        )
        settings = await runner._hmwa_hygiene_settings(source, "key")
        plan = await runner._hmwa_hygiene_plan(
            settings, _history(), SimpleNamespace(session_id="tools", last_prompt_tokens=100_000), "key",
            incoming_message={"role": "user", "content": "next"},
        )

        assert calls == []
        assert plan.needs_compress is supplied_client
        assert (plan.approx_tokens >= 50_000) is supplied_client
    finally:
        providers._REGISTRY.pop(profile.name, None)
        providers._SOURCES.pop(profile.name, None)
        providers._PROVIDER_LIST_CACHE = None
