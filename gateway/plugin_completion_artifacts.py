"""Validate and retain plugin completion images for transcript replay."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import string
import tempfile

from gateway.plugin_completion_store import private_directory
from hermes_state_runtime import RuntimeStoreError

_EXTENSIONS = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/gif': '.gif', 'image/webp': '.webp'}


def request_descriptor(event_id, text, artifacts):
    if (not isinstance(event_id, str) or not event_id or len(event_id) > 256
            or not isinstance(text, str) or len(text) > 65536
            or not isinstance(artifacts, (list, tuple)) or len(artifacts) > 10):
        raise RuntimeStoreError('invalid_params')
    normalized = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) - {'path', 'mime_type', 'name'}:
            raise RuntimeStoreError('invalid_params')
        path, mime = artifact.get('path'), artifact.get('mime_type')
        name = artifact.get('name')
        if (not isinstance(path, str) or not Path(path).is_absolute() or '\x00' in path
                or mime not in _EXTENSIONS or (name is not None and not isinstance(name, str))):
            raise RuntimeStoreError('invalid_params')
        normalized.append({'path': path, 'mime_type': mime, 'name': name})
    if not text.strip() and not normalized:
        raise RuntimeStoreError('invalid_params')
    return json.dumps({'text': text, 'artifacts': normalized}, sort_keys=True, separators=(',', ':'))


def _matches_mime(header, mime):
    return {'image/png': header.startswith(b'\x89PNG\r\n\x1a\n'),
            'image/jpeg': header.startswith(b'\xff\xd8\xff'),
            'image/gif': header.startswith((b'GIF87a', b'GIF89a')),
            'image/webp': header.startswith(b'RIFF') and header[8:12] == b'WEBP'}[mime]


def snapshot(home, session_key, artifact, remaining):
    from gateway.platforms.base import validate_media_delivery_path
    approved = validate_media_delivery_path(artifact['path'], session_key=session_key)
    if approved is None:
        raise RuntimeStoreError('permission_denied')
    root = private_directory(home, 'cache/documents/plugin-completions')
    descriptor = os.open(approved, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    temporary = None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > remaining:
            raise RuntimeStoreError('invalid_params')
        with os.fdopen(descriptor, 'rb', closefd=False) as source:
            header = source.read(16)
            if not _matches_mime(header, artifact['mime_type']):
                raise RuntimeStoreError('invalid_params')
            source.seek(0)
            output_descriptor, temporary = tempfile.mkstemp(prefix='.capture-', dir=root)
            digest, size = hashlib.sha256(), 0
            with os.fdopen(output_descriptor, 'wb') as output:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > remaining:
                        raise RuntimeStoreError('invalid_params')
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        filename = artifact.get('name') or Path(approved).name
        filename = re.sub(r'[^\w.-]', '_', Path(filename).stem)[:100].strip('.') or 'image'
        folder = private_directory(home, 'cache/documents/plugin-completions/'+digest.hexdigest())
        path = folder/(filename+_EXTENSIONS[artifact['mime_type']])
        if path.is_symlink():
            raise RuntimeStoreError('permission_denied')
        os.replace(temporary, path)
        from utils import fsync_directory
        fsync_directory(folder)
        fsync_directory(root)
        return str(path), size
    finally:
        os.close(descriptor)
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def completion_body(home, ticket, request):
    from gateway.platforms.base import get_inbound_media_max_bytes
    value = json.loads(request)
    # Text cannot smuggle an attachment past the artifact validator. The host is
    # the only writer of MEDIA directives in a completion note.
    # Numeric character entities remain text nodes, including with math plugins
    # that reinterpret CommonMark's backslash escapes before Markdown parsing.
    caption = ''.join(f'&#{ord(character)};' if character in string.punctuation else character
                      for character in value['text'])
    remaining, paths = get_inbound_media_max_bytes(), []
    for artifact in value['artifacts']:
        path, size = snapshot(home, ticket['session_key'], artifact, remaining)
        paths.append(path)
        remaining -= size
    return '\n\n'.join(filter(None, (caption, *('MEDIA:'+path for path in paths))))
