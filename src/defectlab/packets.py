"""Offline, explicit text evidence bundles. This module never sends a request."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .common import hash_file, safe_path, write_json

_ALLOWED_ROOTS = {'docs', 'src', 'tests', 'configs', 'scripts'}
_ALLOWED_FILES = {'README.md', 'AGENTS.md', 'pyproject.toml'}
_ALLOWED_SUFFIXES = {'.md', '.py', '.json', '.toml'}


def prepare_packet(repo_root, specification, output_dir):
    """Copy an allowlisted text packet to a new private run directory.

    A packet is prepared, not cleared for disclosure. A person/conductor must
    inspect its exact file list and content before any external transmission.
    """
    root = Path(repo_root).resolve()
    spec_path = Path(specification).resolve()
    spec_raw = spec_path.read_bytes()
    spec = json.loads(spec_raw)
    if spec.get('schema_version') != 1:
        raise ValueError('unsupported packet schema')
    for field in ('task_id', 'question', 'rules', 'acceptance_criteria', 'output_contract', 'files'):
        if not spec.get(field):
            raise ValueError(f'packet requires {field}')
    if not isinstance(spec['files'], list) or not all(isinstance(x, str) for x in spec['files']):
        raise ValueError('files must be an explicit list of relative text paths')
    if len(set(spec['files'])) != len(spec['files']):
        raise ValueError('duplicate packet file')
    output = Path(output_dir).resolve()
    if not output.is_relative_to(root / 'runs') or output == root / 'runs':
        raise ValueError('packet output must be a new directory under runs/')
    if output.exists():
        raise ValueError('packet output already exists; preserve prior attempts')
    entries, content = [], []
    # Validate and read every source before creating any output.
    for name in spec['files']:
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                or (relative.parts[0] not in _ALLOWED_ROOTS and name not in _ALLOWED_FILES)
                or relative.suffix not in _ALLOWED_SUFFIXES):
            raise ValueError(f'file outside permitted text source locations: {name}')
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f'symlink source is not allowed: {name}')
        source = safe_path(root, name)
        raw = source.read_bytes()
        decoded = raw.decode('utf-8')
        if b'\x00' in raw:
            raise ValueError(f'binary content is not permitted: {name}')
        entries.append({'path': relative.as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
        content.append((relative.as_posix(), raw, decoded))
    manifest = {
        'schema_version': 1, 'task_id': spec['task_id'],
        'specification_sha256': hashlib.sha256(spec_raw).hexdigest(), 'files': entries,
        'state': 'prepared', 'external_transmission': 'not_performed',
        'outbound_review': 'required', 'observed_model': None, 'observed_effort': None,
        'limitations': ['Explicit text allowlist is not a secret detector or disclosure approval.',
                        'No browser, model, upload or independent review is invoked.'],
    }
    prompt = [f"# {spec['task_id']}", '', spec['question'], '']
    for label, field in [('Rules', 'rules'), ('Acceptance criteria', 'acceptance_criteria'),
                         ('Required output', 'output_contract')]:
        prompt.extend([f'## {label}', '', str(spec[field]), ''])
    prompt.extend(['## Evidence', '', 'Use the attached source files and manifest. Cite path and line.',
                   'Separate observation, inference and unknown. Do not claim to have executed code.', ''])
    evidence = ['# Source evidence', '']
    for name, _, decoded in content:
        evidence.extend([f'## {name}', ''])
        evidence.extend(f'{i:5}: {line}' for i, line in enumerate(decoded.splitlines(), 1))
        evidence.append('')
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, raw, _ in content:
            destination = output / 'sources' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        (output / 'specification.json').write_bytes(spec_raw)
        (output / 'prompt.md').write_text('\n'.join(prompt), encoding='utf-8')
        (output / 'evidence.md').write_text('\n'.join(evidence), encoding='utf-8')
        manifest['artifacts'] = [
            {'path': name, 'sha256': hash_file(output / name)}
            for name in ('prompt.md', 'evidence.md', 'specification.json')
        ]
        write_json(output / 'manifest.json', manifest)
    except BaseException:
        shutil.rmtree(output)
        raise
    return manifest
