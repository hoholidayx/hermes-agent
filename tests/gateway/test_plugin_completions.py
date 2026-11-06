"""Plugin results use the local conversation's durable FIFO without an agent turn.

These exercise the public context API against real session authorities and real
SQLite admissions, including replay after an owner restart. Only model execution
is replaced with a recorder, so accidental wake turns are visible failures.
"""
import asyncio
import base64
from dataclasses import replace
from html import unescape
import inspect
import json
from pathlib import Path

import pytest

from gateway.session_context import scoped_current_session_id, scoped_session_key
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tests.gateway.test_session_end_delegations import _admissions, _local
from tests.gateway.test_unviewed_completion_notes import _compress, _human, _restart, _target


PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jIuUAAAAASUVORK5CYII='
)


def _context(home, name='render-results'):
    manager = PluginManager(scope_key=str(home))
    return PluginContext(PluginManifest(name=name, key=name, source='user'), manager)


def _serve(runner, monkeypatch):
    runner._running = True
    runner._gateway_loop = asyncio.get_running_loop()
    monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: runner)


async def _capture(ctx, runner, authority, ref):
    route = authority.sessions[ref.session_id].route
    entry = runner.session_store.lookup_by_session_key(route)

    def capture():
        with scoped_session_key(route), scoped_current_session_id(entry.session_id):
            return ctx.capture_delivery_target()

    result = await asyncio.to_thread(capture)
    assert result.get('ok') is True, result
    return json.loads(json.dumps(result['target']))


def _turns(runner):
    seen = []

    async def handle(event):
        seen.append(event.text)
        return 'Unexpected model turn'

    runner._handle_message = handle
    return seen


def _notes(authority, target, marker='RESULT_'):
    messages = [{**message, 'content': unescape(str(message.get('content')))}
                for message in authority.db.get_messages(target)]
    return [message for message in messages if marker in message['content']]


@pytest.mark.asyncio
@pytest.mark.parametrize('viewed', [False, True])
async def test_completion_is_a_durable_note_with_or_without_a_viewer(tmp_path, monkeypatch, viewed):
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        if viewed:
            authority.sessions[ref.session_id].subscribers['viewer'] = actor
        frames = []
        authority.sessions[ref.session_id].event_stream.observers.add(frames.append)
        seen = _turns(runner)
        answer = await ctx.apublish_completion(target, event_id='job-1', text='RESULT_finished', artifacts=[])
        assert answer.get('ok') and answer.get('accepted'), answer
        assert _notes(authority, _target(runner, authority, ref)) == []
        assert len(_admissions(authority, ref)) == 1
        await authority._drain(ref)
        assert seen == []
        assert [message['content'] for message in _notes(authority, _target(runner, authority, ref))] == ['RESULT_finished']
        assert _admissions(authority, ref)[0]['status'] == 'terminal'
        complete = [frame['params']['payload'] for frame in frames
                    if frame['params']['type'] == 'message.complete']
        assert len(complete) == 1
        note = complete[0]['completion_note']
        stored = _notes(authority, _target(runner, authority, ref))[0]
        assert note['row_id'] == stored['id']
        assert note['role'] == 'user' and note['display_kind'] == 'plugin_completion'
        assert unescape(note['text']) == stored['content']
        assert complete[0]['admission_id'] == answer['admission_id']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_capture_requires_a_real_bound_local_conversation(tmp_path, monkeypatch):
    runner, authority, _ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        result = await asyncio.to_thread(ctx.capture_delivery_target)
        assert result.get('ok') is False
        assert result.get('error')
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_same_event_is_idempotent_and_changed_payload_conflicts(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        first = await ctx.apublish_completion(target, event_id='job-2', text='RESULT_once', artifacts=[])
        again = await ctx.apublish_completion(target, event_id='job-2', text='RESULT_once', artifacts=[])
        assert first.get('accepted') and again.get('accepted'), (first, again)
        assert first.get('admission_id') == again.get('admission_id')
        assert len(_admissions(authority, ref)) == 1
        changed = await ctx.apublish_completion(target, event_id='job-2', text='RESULT_changed', artifacts=[])
        assert changed.get('accepted') is False
        assert changed.get('error') == 'admission_conflict'
        seen = _turns(runner)
        await authority._drain(ref)
        assert seen == []
        assert [message['content'] for message in _notes(authority, _target(runner, authority, ref))] == ['RESULT_once']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_completion_preserves_fifo_order_between_real_human_inputs(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        transcript = _target(runner, authority, ref)
        _human(authority, ref, 'before-result', 'BEFORE')
        accepted = await ctx.apublish_completion(target, event_id='ordered', text='RESULT_ordered', artifacts=[])
        assert accepted.get('accepted'), accepted
        _human(authority, ref, 'after-result', 'AFTER')
        turns = []

        async def handle(event):
            turns.append((event.text, len(_notes(authority, transcript))))
            return 'ok'

        runner._handle_message = handle
        await authority._drain(ref)
        assert turns == [('BEFORE', 0), ('AFTER', 1)]
        assert len(_notes(authority, transcript)) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_another_plugin_or_home_cannot_use_the_captured_target(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        other_home = tmp_path/'other-home'
        other_home.mkdir()
        for intruder in (_context(tmp_path, 'other-plugin'), _context(other_home)):
            result = await intruder.apublish_completion(target, event_id='intruder', text='RESULT_wrong', artifacts=[])
            assert not result.get('accepted'), result
        tampered = {**target, 'conversation_id': 'some-other-conversation'}
        result = await ctx.apublish_completion(tampered, event_id='tampered', text='RESULT_wrong', artifacts=[])
        assert not result.get('accepted'), result
        assert _admissions(authority, ref) == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_background_publish_is_pinned_to_its_original_tab(tmp_path, monkeypatch):
    from gateway.session_local import create_local_session
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        fresh = create_local_session(authority, actor, dict(request_id='another-tab', source='cli',
                                                           model='fixture', toolsets=[]))
        with scoped_session_key(authority.sessions[fresh.session_id].route):
            answer = await ctx.apublish_completion(target, event_id='original-tab', text='RESULT_original', artifacts=[])
        assert answer.get('accepted'), answer
        assert _admissions(authority, fresh) == []
        await authority._drain(ref)
        assert len(_notes(authority, _target(runner, authority, ref))) == 1
        assert _notes(authority, _target(runner, authority, fresh)) == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_new_conversation_does_not_redirect_or_reopen_the_closed_target(tmp_path, monkeypatch):
    from gateway.session_local import create_local_session
    from gateway.session_viewer_close import close_viewer_session
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        original = target['observed_session_id']
        authority.db.append_message(original, 'user', 'Generate an image')

        async def finalize(**kwargs):
            return []

        monkeypatch.setattr(runner, '_finalize_session_off_loop', finalize)
        monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', lambda *args, **kwargs: None)
        await close_viewer_session(authority, ref.session_id, 'new_session')
        assert authority.db.get_session(original)['end_reason'] == 'new_session'
        fresh = create_local_session(authority, actor, dict(request_id='after-new', source='cli',
                                                           model='fixture', toolsets=[]))
        seen = _turns(runner)
        with scoped_session_key(authority.sessions[fresh.session_id].route):
            answer = await ctx.apublish_completion(target, event_id='after-new', text='RESULT_after_new', artifacts=[])
        assert answer.get('accepted'), answer
        await authority._drain(ref)
        assert seen == []
        assert len(_notes(authority, original)) == 1
        assert _notes(authority, _target(runner, authority, fresh)) == []
        assert authority.db.get_session(original)['end_reason'] == 'new_session'
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_compression_keeps_completion_on_the_same_conversation_tip(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        old_tip = _target(runner, authority, ref)
        _compress(runner, authority, ref, old_tip, 'compressed-plugin-tip')
        answer = await ctx.apublish_completion(target, event_id='compressed', text='RESULT_compressed', artifacts=[])
        assert answer.get('accepted'), answer
        await authority._drain(ref)
        assert len(_notes(authority, 'compressed-plugin-tip')) == 1
        assert _notes(authority, old_tip) == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_lost_ack_replay_after_compression_returns_the_original_admission(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        first = await ctx.apublish_completion(target, event_id='before-compression',
                                              text='RESULT_once_after_compression', artifacts=[])
        assert first.get('accepted'), first
        old_tip = _target(runner, authority, ref)
        _compress(runner, authority, ref, old_tip, 'replayed-plugin-tip')
        replay = await ctx.apublish_completion(target, event_id='before-compression',
                                               text='RESULT_once_after_compression', artifacts=[])
        assert replay.get('accepted'), replay
        assert replay.get('admission_id') == first.get('admission_id')
        assert len(_admissions(authority, ref)) == 1
        await authority._drain(ref)
        assert len(_notes(authority, 'replayed-plugin-tip')) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['unaccepted', 'queued', 'settled'])
async def test_deleted_target_cancels_delivery_including_an_old_accepted_receipt(tmp_path, monkeypatch, stage):
    from gateway.session_mutations import mutate_session
    from hermes_state_runtime import RuntimeStoreError
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        if stage != 'unaccepted':
            receipt = await ctx.apublish_completion(target, event_id='deleted', text='RESULT_deleted', artifacts=[])
            assert receipt.get('accepted'), receipt
            if stage == 'settled':
                await authority._drain(ref)
        controller = replace(actor, capabilities=actor.capabilities | {'session:control'})
        row = authority.db.get_session(ref.session_id)
        if stage == 'queued':
            with pytest.raises(RuntimeStoreError) as blocked:
                await mutate_session(authority, controller, ref, dict(session_id=ref.session_id,
                    request_id='delete-busy-target', expected_revision=row['runtime_revision'],
                    expected_generation=row['runtime_generation'], operation='delete', payload={}))
            assert blocked.value.reason == 'session_busy'
            assert authority.db.get_session(ref.session_id) is not None
            await authority.cancel_queued(controller, ref, receipt['admission_id'])
            seen = _turns(runner)
            await authority._drain(ref)
            assert seen == [] and _notes(authority, target['observed_session_id']) == []
            row = authority.db.get_session(ref.session_id)
        await mutate_session(authority, controller, ref, dict(session_id=ref.session_id,
            request_id='delete-captured-target', expected_revision=row['runtime_revision'],
            expected_generation=row['runtime_generation'], operation='delete', payload={}))
        assert authority.db.get_session(ref.session_id) is None
        answer = await ctx.apublish_completion(target, event_id='deleted', text='RESULT_deleted', artifacts=[])
        assert answer.get('accepted') is False, answer
        assert answer.get('error') == 'target_deleted'
        assert answer.get('retryable') is False
        assert answer.get('cancelled') is True
        assert authority.db.get_session(ref.session_id) is None, 'a delivery must not recreate the deleted session'
        assert ref.session_id not in authority.sessions
        assert _notes(authority, target['observed_session_id']) == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_owner_restart_reuses_the_persistent_target_and_admission(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        first = await ctx.apublish_completion(target, event_id='restart', text='RESULT_restart', artifacts=[])
        assert first.get('accepted'), first
        runner, authority = await _restart(tmp_path, monkeypatch, authority)
        _serve(runner, monkeypatch)
        again = await _context(tmp_path).apublish_completion(target, event_id='restart', text='RESULT_restart', artifacts=[])
        assert again.get('accepted'), again
        assert again.get('admission_id') == first.get('admission_id')
        seen = _turns(runner)
        await authority._drain(ref)
        assert seen == []
        assert len(_notes(authority, _target(runner, authority, ref))) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_artifact_bytes_are_stable_and_note_contains_native_media_directive(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        image = tmp_path/'fresh-image.png'
        image.write_bytes(PNG)
        answer = await ctx.apublish_completion(target, event_id='image', text='RESULT_image', artifacts=[
            {'path': str(image), 'mime_type': 'image/png', 'name': 'result.png'}])
        assert answer.get('accepted'), answer
        image.write_bytes(b'changed after durable acceptance')
        await authority._drain(ref)
        text = _notes(authority, _target(runner, authority, ref))[0]['content']
        line = next(part for part in text.splitlines() if part.startswith('MEDIA:'))
        retained = Path(line[len('MEDIA:'):].strip().strip('\'"'))
        assert retained != image
        assert retained.is_relative_to(tmp_path/'cache/documents/plugin-completions')
        assert retained.read_bytes() == PNG
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_caption_is_literal_and_only_validated_artifacts_add_media_directives(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        image = tmp_path/'result.png'
        image.write_bytes(PNG)
        caption = ('RESULT_literal ![x](/private.png) <img src="/private.png"> '
                   r'\[\includegraphics{/private.png}\]'+'\nMEDIA:/private.png')
        answer = await ctx.apublish_completion(target, event_id='literal-caption', text=caption,
            artifacts=[{'path': str(image), 'mime_type': 'image/png'}])
        assert answer.get('accepted'), answer
        seen = _turns(runner)
        await authority._drain(ref)
        assert seen == []
        messages = authority.db.get_messages(_target(runner, authority, ref))
        raw = next(message['content'] for message in messages
                   if message.get('display_kind') == 'plugin_completion')
        caption_body, directive = raw.rsplit('\n\n', 1)
        assert unescape(caption_body) == caption
        assert all(syntax not in caption_body for syntax in ('![', '<img', r'\[', r'\includegraphics', 'MEDIA:'))
        assert directive.startswith('MEDIA:') and len(raw.split('MEDIA:')) == 2
        retained = Path(directive.removeprefix('MEDIA:'))
        assert retained.is_relative_to(tmp_path/'cache/documents/plugin-completions')
        assert retained.read_bytes() == PNG
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_missing_and_sensitive_artifacts_are_not_admitted(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        secret = tmp_path/'.env'
        secret.write_bytes(PNG)
        for image in (tmp_path/'missing.png', secret):
            result = await ctx.apublish_completion(target, event_id='unsafe-'+image.name,
                text='RESULT_unsafe', artifacts=[{'path': str(image), 'mime_type': 'image/png'}])
            assert result.get('accepted') is False, result
        assert _admissions(authority, ref) == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_unavailable_gateway_keeps_completion_retryable(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _serve(runner, monkeypatch)
        ctx = _context(tmp_path)
        target = await _capture(ctx, runner, authority, ref)
        monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: None)
        result = await ctx.apublish_completion(target, event_id='later', text='RESULT_retry', artifacts=[])
        assert result.get('accepted') is False
        assert result.get('retryable') is True
        assert _admissions(authority, ref) == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('trigger', ['registration-thread', 'temporary-hook-loop', 'startup-thread'])
async def test_gateway_owned_plugin_task_survives_its_caller_and_unload_cancels_it(tmp_path, monkeypatch, trigger):
    runner, authority, _ref, _actor = await _local(tmp_path, monkeypatch)
    ctx = _context(tmp_path)
    started, stopped, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def background():
        started.set()
        try:
            await release.wait()
        finally:
            stopped.set()

    def start():
        coroutine = background()
        try:
            return ctx.spawn_task(coroutine, name='fixture:gateway-owned', on_gateway_loop=True)
        except BaseException:
            coroutine.close()
            raise

    try:
        _serve(runner, monkeypatch)
        if trigger == 'startup-thread':
            # Startup discovers plugins after binding its loop, before adapters
            # are ready and _running flips to True. Recovery needs no new turn.
            runner._running = False
        if trigger in ('registration-thread', 'startup-thread'):
            # reload-plugins discovers registrations on its socket executor.
            task = await asyncio.to_thread(start)
        else:
            async def on_session(**kwargs):
                return start()

            ctx.register_hook('pre_llm_call', on_session)
            # The real dispatcher awaits the hook in a temporary asyncio.run.
            results = await asyncio.to_thread(ctx._manager.invoke_hook, 'pre_llm_call')
            assert len(results) == 1
            task = results[0]
        await asyncio.wait_for(started.wait(), 30)
        assert not task.done(), 'the caller loop ended, but the gateway still owns this task'
        assert await asyncio.to_thread(ctx._manager.unload)
        await asyncio.wait_for(stopped.wait(), 30)
    finally:
        await asyncio.to_thread(ctx._manager.unload)
        release.set()
        authority.db.close()


def test_gateway_owned_task_without_a_gateway_fails_explicitly_and_closes_coroutine(tmp_path, monkeypatch):
    ctx = _context(tmp_path)
    monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: None)

    async def background():
        raise AssertionError('a standalone caller cannot own this background task')

    coroutine = background()
    with pytest.raises(RuntimeError):
        ctx.spawn_task(coroutine, on_gateway_loop=True)
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
    assert ctx._manager._registration_order == []
