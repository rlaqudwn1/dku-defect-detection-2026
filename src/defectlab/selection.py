"""Freeze the concrete files and tolerance evidence used for a model selection."""
from pathlib import Path

from .common import hash_file, load_json, write_json


def select(validation_path, test_path, reproduction_path, output_path, reason):
    validation, test = load_json(validation_path), load_json(test_path)
    reproduction = load_json(reproduction_path)
    if not reason or reproduction.get('passed') is not True:
        raise ValueError('selection requires a reason and passed reproduction evidence')
    if validation.get('kind') != 'validation' or test.get('kind') != 'test':
        raise ValueError('selection requires validation and test prediction manifests')
    if reproduction.get('reference_sha256') != hash_file(validation_path):
        raise ValueError('reproduction does not reference selected validation bytes')
    for key in ('dataset_hash', 'split_hash'):
        if not validation.get(key) or validation[key] != test.get(key):
            raise ValueError('selected validation/test provenance mismatch: ' + key)
    if validation.get('recipe'):
        left, right = validation['recipe'], test.get('recipe', {})
        left_members = [(m['checkpoint_sha256'], m['protocol_hash']) for m in left['members']]
        right_members = [(m['checkpoint_sha256'], m['protocol_hash']) for m in right.get('members', [])]
        if left_members != right_members or left['weights'] != right.get('weights'):
            raise ValueError('selected ensemble composition differs between validation and test')
        checkpoints = [(m['checkpoint_path'], m['checkpoint_sha256']) for m in left['members']]
    else:
        for key in ('checkpoint_sha256', 'protocol_hash'):
            if not validation.get(key) or validation[key] != test.get(key):
                raise ValueError('selected model validation/test mismatch: ' + key)
        checkpoints = [(validation['checkpoint_path'], validation['checkpoint_sha256'])]
    if Path(output_path).exists():
        raise ValueError('selection output already exists; preserve previous selections')
    for path, declared_hash in checkpoints:
        if hash_file(path) != declared_hash:
            raise ValueError('selected checkpoint bytes differ from evaluated checkpoint')
    evidence = []
    for path in [validation_path, test_path, reproduction_path, *(p for p, _ in checkpoints)]:
        evidence.append({'path': str(Path(path).resolve()), 'sha256': hash_file(path)})
    selected = {'schema_version': 1, 'state': 'selected-local', 'reason': reason,
                'evidence': evidence, 'validation_auroc': validation.get('auroc'),
                'dataset_hash': validation['dataset_hash'], 'split_hash': validation['split_hash'],
                'platform_submitted': False, 'fresh_environment_reproduction': None,
                'limitations': ['Local fixed-weight reproduction is distinct from a new machine or GPU rehearsal.']}
    write_json(output_path, selected)
    return selected
