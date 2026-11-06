"""Plain-data completion API; capture in a turn, publish after it ends."""
from __future__ import annotations

import asyncio
import concurrent.futures


class PluginDeliveryCtxMixin:
    """Generic plugin posts and durable local completions on the public context."""

    def post_message(self, text: str):
        """Post to the bound conversation; edit through the returned message handle."""
        from hermes_cli.plugin_messages import post
        return post(self, text)

    async def apost_message(self, text: str):
        """Post from async code on the gateway loop."""
        from hermes_cli.plugin_messages import apost
        return await apost(self, text)

    def capture_delivery_target(self) -> dict:
        """Capture this local conversation as a durable host-issued reference."""
        return capture(self)

    async def apublish_completion(self, target: dict, *, event_id: str, text: str = '',
                                  artifacts: list | tuple = ()) -> dict:
        """Durably queue a local result and optional images, without a model turn.

        Stable event IDs deduplicate retries. Only targets this plugin captured
        are valid; accepted confirms admission rather than a connected viewer.
        """
        return await publish(self, target, event_id=event_id, text=text, artifacts=artifacts)


def capture(ctx):
    from gateway.plugin_completions import capture as capture_target, refusal
    from gateway.plugin_messages import bound_route
    from gateway.session_context import bound_session_key, get_session_env
    from hermes_cli.plugin_messages import _gateway_loop
    key = bound_session_key()
    if not key:
        return refusal('no_conversation')
    route, observed = bound_route(), get_session_env('HERMES_SESSION_ID', '')
    runner, loop = _gateway_loop()
    if runner is None:
        return refusal('gateway_unavailable', retryable=True)
    try:
        current = asyncio.get_running_loop()
    except RuntimeError:
        current = None
    if current is loop:
        return refusal('on_event_loop')

    async def capture_on_owner():
        # The caller's identity was frozen before hopping to the owner loop.
        # Its live route cannot be replaced while capture checks and issues it.
        return capture_target(ctx, runner, key, route, observed)

    future = asyncio.run_coroutine_threadsafe(capture_on_owner(), loop)
    try:
        return future.result(timeout=30)
    except concurrent.futures.TimeoutError:
        return refusal('capture_unconfirmed', retryable=True)


async def publish(ctx, target, *, event_id, text, artifacts):
    from gateway.plugin_completions import publish as publish_target, refusal
    from hermes_cli.plugin_messages import _gateway_loop
    runner, loop = _gateway_loop()
    if runner is None:
        return refusal('gateway_unavailable', retryable=True)
    operation = publish_target(ctx, runner, target, event_id=event_id, text=text, artifacts=artifacts)
    if asyncio.get_running_loop() is loop:
        return await operation
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(operation, loop))
