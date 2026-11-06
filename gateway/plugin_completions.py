"""Local plugin results admitted as durable notes, never as agent turns."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3

from gateway.plugin_completion_store import CompletionStore, admission_receipt
from hermes_state_runtime import RuntimeStoreError

PRODUCER = 'plugin-completion'
DISPLAY_KIND = 'plugin_completion'
_PERMANENT = frozenset({'invalid_params', 'permission_denied', 'profile_mismatch',
                        'not_found', 'admission_conflict', 'target_deleted'})


def refusal(code, *, retryable=False):
    return {'ok': False, 'accepted': False, 'error': code, 'retryable': retryable,
            **({'cancelled': True} if code == 'target_deleted' else {})}


def _authority(ctx, runner):
    from gateway.session_authorities import authority_for_home
    authority = authority_for_home(runner, ctx._manager.home_path)
    if authority is None:
        raise RuntimeStoreError('storage_unavailable')
    if Path(authority.db.db_path).resolve().parent != Path(ctx._manager.home_path).resolve():
        raise RuntimeStoreError('profile_mismatch')
    return authority


def capture(ctx, runner, session_key, route, observed_session_id):
    from gateway import plugin_messages
    from gateway.session_detached import route_entry
    try:
        authority = _authority(ctx, runner)
        target, error = plugin_messages.resolve(runner, session_key=session_key, route=route,
                                                plugin_home=ctx._manager.home_path, plugin_id=ctx.plugin_id)
        if target is None:
            return refusal((error or {}).get('error', 'no_conversation'))
        if target.live is None or target.platform != 'local':
            return refusal('unsupported_surface')
        entry = route_entry(authority, session_key)
        if entry is None:
            raise RuntimeStoreError('not_found')
        owner = authority.logical_owner(entry.session_id)
        if observed_session_id and authority.logical_owner(observed_session_id) != owner:
            raise RuntimeStoreError('admission_conflict')
        ticket = CompletionStore(ctx._manager.home_path).issue(ctx.plugin_id, owner, session_key, entry.session_id)
        return {'ok': True, 'target': ticket, 'conversation_id': owner,
                'observed_session_id': entry.session_id}
    except RuntimeStoreError as exc:
        return refusal(exc.reason, retryable=exc.reason not in _PERMANENT)
    except (OSError, sqlite3.Error):
        return refusal('storage_unavailable', retryable=True)


def _require_existing(authority, ticket):
    from hermes_state_mutation_retirement import retired_session
    owner = ticket['conversation_id']
    if retired_session(authority.db, owner) or authority.db.get_session(owner) is None:
        raise RuntimeStoreError('target_deleted')


def _local_target(authority, ticket):
    from gateway.config import Platform
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_detached import route_entry
    _require_existing(authority, ticket)
    owner = ticket['conversation_id']
    ref = restore_local_session(authority, owner)
    live = authority.sessions[ref.session_id]
    entry = route_entry(authority, live.route)
    if (live.source.platform != Platform.LOCAL or entry is None
            or authority.logical_owner(entry.session_id) != owner):
        raise RuntimeStoreError('admission_conflict')
    adapter = authority.runner._adapter_for_source(live.source)
    if adapter is None or not adapter.authorize_source(live.source):
        raise RuntimeStoreError('permission_denied')
    return live, entry, adapter


async def _prepared(ctx, store, ticket, event_id, request):
    from gateway.plugin_completion_artifacts import completion_body
    event = await asyncio.to_thread(store.prepared, ticket['token'], event_id, request)
    if event is not None:
        return event
    body = await asyncio.to_thread(completion_body, Path(ctx._manager.home_path), ticket, request)
    return await asyncio.to_thread(store.prepare, ticket['token'], event_id, request, body)


async def publish(ctx, runner, target, *, event_id, text, artifacts):
    from gateway.session_authorities import owner_scope
    from gateway.plugin_completion_artifacts import request_descriptor
    ticket = authority = None
    try:
        authority = _authority(ctx, runner)
        store = CompletionStore(ctx._manager.home_path)
        ticket = await asyncio.to_thread(store.target, ctx.plugin_id, target)
        _require_existing(authority, ticket)
        request = request_descriptor(event_id, text, artifacts)
        with owner_scope(authority):
            prepared = await _prepared(ctx, store, ticket, event_id, request)
            _require_existing(authority, ticket)
            if prepared['receipt'] is not None:
                return prepared['receipt']
            identity = json.dumps([PRODUCER, ctx.plugin_id, ticket['conversation_id'], event_id],
                                  separators=(',', ':'))
            receipt = await asyncio.to_thread(admission_receipt, authority, ticket, identity, prepared['body'])
            if receipt is None:
                label = text.strip()[:2000] or 'Plugin result'
                receipt = await _admit(authority, ticket, identity, prepared['body'], label)
            await asyncio.to_thread(store.acknowledge, ticket['token'], event_id, receipt)
            return receipt
    except RuntimeStoreError as exc:
        code = exc.reason
        if code == 'not_found' and ticket is not None and authority is not None:
            try:
                _require_existing(authority, ticket)
            except RuntimeStoreError:
                code = 'target_deleted'
        return refusal(code, retryable=code not in _PERMANENT)
    except (FileNotFoundError, ValueError):
        return refusal('invalid_artifact')
    except (OSError, sqlite3.Error):
        return refusal('storage_unavailable', retryable=True)


async def _admit(authority, ticket, identity, body, label):
    from gateway.platforms.event import MessageEvent, MessageType
    live, entry, adapter = _local_target(authority, ticket)
    event = MessageEvent(text=body, message_type=MessageType.TEXT, source=live.source, internal=True,
                         allow_gateway_control=False, message_id=identity,
                         metadata={'gateway_session_key': live.route, 'gateway_session_id': entry.session_id,
                                   'display_kind': DISPLAY_KIND, 'display_text': label})
    receipt = await authority.admit_automation(adapter, event, identity)
    return {'ok': True, 'accepted': True, 'admission_id': receipt.admission_id,
            'state': receipt.status, 'conversation_id': ticket['conversation_id']}
