"""Plain completion receipts cross a real external plugin-host connection.

The tool's capture callback retains the original parent's session context. Its
later async publish runs with no bound caller and still targets that conversation.
"""
import asyncio
from html import unescape
import json
import os

import pytest

from gateway.session_context import bound_session_key, scoped_current_session_id, scoped_session_key
from hermes_cli.plugins import PluginManager
from tests.gateway.test_session_end_delegations import _admissions, _local
from tests.gateway.test_unviewed_completion_notes import _target
from tests.hermes_cli.test_plugin_host import _home_with_plugins
from tests.waiting import async_wait_until


PLUGIN = '''
import asyncio, json, os
from pathlib import Path

TARGET = None

def register(ctx):
    async def background():
        status = Path(__file__).with_name('task-status.txt')
        status.write_text('started')
        try:
            await asyncio.Event().wait()
        finally:
            status.write_text('cancelled')

    ctx.spawn_task(background(), name='completion-host:owned', on_gateway_loop=True)

    def capture(args, **kw):
        global TARGET
        result = ctx.capture_delivery_target()
        TARGET = result.get('target')
        return json.dumps({'pid': os.getpid(), 'captured': result})

    async def publish(args, **kw):
        receipt = await ctx.apublish_completion(TARGET, event_id='external-host-event',
            text='RESULT_external_host', artifacts=[])
        return json.dumps({'pid': os.getpid(), 'receipt': receipt})

    for name, handler, is_async in (('completion_host_capture', capture, False),
                                     ('completion_host_publish', publish, True)):
        ctx.register_tool(name=name, toolset='completion_host',
            schema={'name': name, 'description': 'Completion test',
                    'parameters': {'type': 'object', 'properties': {}}},
            handler=handler, is_async=is_async)
'''


@pytest.mark.asyncio
@pytest.mark.platforms('any')
async def test_external_host_capture_context_and_awaited_completion_reach_the_real_fifo(tmp_path, monkeypatch):
    from tools.registry import registry
    home = _home_with_plugins(tmp_path, monkeypatch, {'completion-host': PLUGIN})
    runner, authority, ref, _actor = await _local(home, monkeypatch)
    runner._running = True
    runner._gateway_loop = asyncio.get_running_loop()
    monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: runner)
    manager = PluginManager(scope_key=str(home))
    try:
        await asyncio.to_thread(manager.discover_and_load)
        assert manager._plugins['completion-host'].error is None
        task_status = home/'plugins/completion-host/task-status.txt'
        await async_wait_until(task_status.exists, timeout=30, what='host-owned task startup')
        assert task_status.read_text() == 'started'
        route = authority.sessions[ref.session_id].route
        durable = _target(runner, authority, ref)

        def capture():
            with scoped_session_key(route), scoped_current_session_id(durable):
                return json.loads(registry.dispatch('completion_host_capture', {}, scope=manager.scope_key))

        captured = await asyncio.to_thread(capture)
        assert captured['pid'] != os.getpid(), 'the plugin must execute in its own process'
        assert captured['captured'].get('ok'), captured
        assert captured['captured']['target']['observed_session_id'] == durable
        assert bound_session_key() == ''
        published = json.loads(await asyncio.to_thread(
            registry.dispatch, 'completion_host_publish', {}, scope=manager.scope_key))
        assert published['pid'] == captured['pid']
        assert isinstance(published['receipt'], dict), published
        assert published['receipt'].get('ok') and published['receipt'].get('accepted'), published
        assert len(_admissions(authority, ref)) == 1
        turns = []

        async def execute(event):
            turns.append(event.text)
            return 'Unexpected model turn'

        runner._handle_message = execute
        await authority._drain(ref)
        assert turns == []
        notes = [m for m in authority.db.get_messages(durable)
                 if 'RESULT_external_host' in unescape(str(m.get('content')))]
        assert len(notes) == 1
        await asyncio.to_thread(manager.unload)
        await async_wait_until(lambda: task_status.read_text() == 'cancelled', timeout=30,
                               what='host-owned task cancellation on unload')
    finally:
        await asyncio.to_thread(manager.unload)
        host = getattr(manager, '_plugin_host_instance', None)
        if host is not None:
            await asyncio.to_thread(host.shutdown)
        authority.db.close()
