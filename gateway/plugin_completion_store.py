"""Host-issued, profile-local delivery tickets and immutable completion receipts."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import uuid

from hermes_state_runtime import RuntimeStoreError


def private_directory(home: Path, relative: str) -> Path:
    from hermes_constants import mkdir_under_hermes_home
    home = home.resolve()
    folder = home / relative
    for component in (folder, *folder.parents):
        if component == home:
            break
        if component.is_symlink():
            raise RuntimeStoreError('permission_denied')
    mkdir_under_hermes_home(folder)
    if not folder.resolve().is_relative_to(home):
        raise RuntimeStoreError('permission_denied')
    return folder


class CompletionStore:
    def __init__(self, home):
        self.home = Path(home).resolve()

    @contextmanager
    def connection(self):
        from hermes_cli.sqlite_util import open_db
        folder = private_directory(self.home, 'state/plugin-completions')
        path = folder / 'receipts.sqlite3'
        if path.is_symlink():
            raise RuntimeStoreError('permission_denied')
        connection = open_db(path, db_label='plugin-completions', synchronous_full=True)
        try:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS targets (
                    token TEXT PRIMARY KEY, plugin_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL, session_key TEXT NOT NULL,
                    observed_session_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS completions (
                    token TEXT NOT NULL, event_id TEXT NOT NULL, request TEXT NOT NULL,
                    body TEXT NOT NULL, receipt TEXT,
                    PRIMARY KEY(token,event_id));
            ''')
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def public_target(row):
        return {'version': 1, 'token': row['token'],
                'conversation_id': row['conversation_id'],
                'observed_session_id': row['observed_session_id']}

    def issue(self, plugin_id, conversation_id, session_key, observed_session_id):
        row = dict(token=uuid.uuid4().hex, plugin_id=plugin_id, conversation_id=conversation_id,
                   session_key=session_key, observed_session_id=observed_session_id)
        with self.connection() as connection:
            connection.execute('INSERT INTO targets VALUES (:token,:plugin_id,:conversation_id,:session_key,:observed_session_id)', row)
        return self.public_target(row)

    def target(self, plugin_id, target):
        if not isinstance(target, dict) or not isinstance(target.get('token'), str):
            raise RuntimeStoreError('permission_denied')
        with self.connection() as connection:
            row = connection.execute('SELECT * FROM targets WHERE token=? AND plugin_id=?',
                                     (target['token'], plugin_id)).fetchone()
        if row is None or target != self.public_target(row):
            raise RuntimeStoreError('permission_denied')
        return dict(row)

    def prepared(self, token, event_id, request):
        with self.connection() as connection:
            row = connection.execute('SELECT * FROM completions WHERE token=? AND event_id=?',
                                     (token, event_id)).fetchone()
        return self._event(row, request)

    @staticmethod
    def _event(row, request):
        if row is None:
            return None
        if row['request'] != request:
            raise RuntimeStoreError('admission_conflict')
        return {'body': row['body'], 'receipt': json.loads(row['receipt']) if row['receipt'] else None}

    def prepare(self, token, event_id, request, body):
        with self.connection() as connection:
            connection.execute('INSERT OR IGNORE INTO completions(token,event_id,request,body) VALUES (?,?,?,?)',
                               (token, event_id, request, body))
            row = connection.execute('SELECT * FROM completions WHERE token=? AND event_id=?',
                                     (token, event_id)).fetchone()
        return self._event(row, request)

    def acknowledge(self, token, event_id, receipt):
        with self.connection() as connection:
            connection.execute('UPDATE completions SET receipt=? WHERE token=? AND event_id=?',
                               (json.dumps(receipt, sort_keys=True), token, event_id))


def admission_receipt(authority, ticket, identity, body):
    """Read a committed replay before route restoration changes physical tips."""
    with authority.db._read_ctx() as connection:
        row = connection.execute('''SELECT admission_id,status,payload_json FROM session_admissions
            WHERE target_session_id=? AND request_id=?''', (ticket['conversation_id'], identity)).fetchone()
    if row is None:
        return None
    if json.loads(row['payload_json']).get('text') != body:
        raise RuntimeStoreError('admission_conflict')
    return {'ok': True, 'accepted': True, 'admission_id': row['admission_id'],
            'state': row['status'], 'conversation_id': ticket['conversation_id']}
