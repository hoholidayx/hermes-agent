"""Ledger-owned plugin tasks on the caller loop or the long-lived gateway loop."""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger('hermes_cli.plugins')


def _gateway_task_loop():
    """Allow boot-time tasks once the owner loop is bound, before adapters are ready."""
    from gateway.run import _gateway_runner_ref

    runner = _gateway_runner_ref()
    loop = getattr(runner, '_gateway_loop', None)
    shutdown = getattr(runner, '_shutdown_event', None)
    if (
        loop is None or loop.is_closed() or not loop.is_running()
        or getattr(runner, '_draining', False)
        or getattr(runner, '_restart_requested', False)
        or getattr(runner, '_stop_task', None) is not None
        or (shutdown is not None and shutdown.is_set())
    ):
        return None
    return loop


def spawn(ctx, coro, *, name=None, on_gateway_loop=False):
    if not asyncio.iscoroutine(coro):
        raise TypeError('spawn_task expects a coroutine')
    task_name = name or f'plugin:{ctx.plugin_id}:task'
    if on_gateway_loop:
        loop = _gateway_task_loop()
        if loop is None:
            coro.close()
            raise RuntimeError('No live gateway loop owns this plugin background task')
        try:
            task = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError:
            coro.close()
            raise
    else:
        task = asyncio.get_running_loop().create_task(coro, name=task_name)
    handle = ctx._track('background_task', task_name, lambda: task.done() or task.cancel())
    task.add_done_callback(lambda _task: handle.dispose())
    logger.debug('Plugin %s spawned supervised task: %s', ctx.manifest.name, task_name)
    return task
