"""Packet boundaries and immutable evidence, without browser or data access."""
import json
from pathlib import Path

import pytest

from defectlab.common import hash_file
from defectlab.packets import prepare_packet


def fixture_spec(root, files):
    path = root / 'spec.json'
    path.write_text(json.dumps({'schema_version': 1, 'task_id': 'pilot',
        'question': 'Which invariant fails?', 'rules': 'No uploads.',
        'acceptance_criteria': 'Preserve groups.', 'output_contract': 'Cite evidence.',
        'files': files}), encoding='utf-8')
    return path


def test_prepared_packet_preserves_sources_and_requires_outbound_review(tmp_path):
    source = tmp_path / 'src' / 'example.py'
    source.parent.mkdir()
    source.write_bytes(b'def example():\n    return 1\n')
    spec = fixture_spec(tmp_path, ['src/example.py'])
    output = tmp_path / 'runs' / 'packet'
    result = prepare_packet(tmp_path, spec, output)
    assert (output / 'specification.json').read_bytes() == spec.read_bytes()
    assert result['specification_sha256'] == hash_file(output / 'specification.json')
    assert result['state'] == 'prepared'
    assert result['external_transmission'] == 'not_performed'
    assert result['outbound_review'] == 'required'
    assert result['files'][0]['sha256'] == hash_file(source)
    assert (output / 'sources' / 'src' / 'example.py').read_bytes() == source.read_bytes()
    assert '    2:     return 1' in (output / 'evidence.md').read_text()
    source.write_text('changed', encoding='utf-8')
    assert (output / 'sources' / 'src' / 'example.py').read_bytes() != source.read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        prepare_packet(tmp_path, spec, output)


@pytest.mark.parametrize('name', ['runs/private.md', '../secret.md', '/tmp/secret.md',
                                 '.env', 'models/model.json', 'src/weights.pt'])
def test_reject_private_or_escaping_sources(tmp_path, name):
    output = tmp_path / 'runs' / 'packet'
    with pytest.raises(ValueError):
        prepare_packet(tmp_path, fixture_spec(tmp_path, [name]), output)
    assert not output.exists()


def test_symlinks_and_invalid_content_leave_no_packet(tmp_path):
    (tmp_path / 'docs').mkdir()
    outside = tmp_path / 'private.md'
    outside.write_text('private', encoding='utf-8')
    link = tmp_path / 'docs' / 'link.md'
    link.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        prepare_packet(tmp_path, fixture_spec(tmp_path, ['docs/link.md']), tmp_path / 'runs' / 'one')
    bad = tmp_path / 'docs' / 'bad.md'
    bad.write_bytes(b'bad\x00data')
    with pytest.raises(ValueError, match='binary'):
        prepare_packet(tmp_path, fixture_spec(tmp_path, ['docs/bad.md']), tmp_path / 'runs' / 'two')
    assert not (tmp_path / 'runs').exists()


def test_output_must_be_private_run_directory(tmp_path):
    source = tmp_path / 'README.md'
    source.write_text('safe', encoding='utf-8')
    with pytest.raises(ValueError, match='under runs'):
        prepare_packet(tmp_path, fixture_spec(tmp_path, ['README.md']), tmp_path / 'docs' / 'packet')
    assert not (tmp_path / 'docs').exists()
    assert Path(source).read_text() == 'safe'
