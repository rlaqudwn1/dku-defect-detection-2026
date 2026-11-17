"""Actual ResNet18 optimizer/checkpoint/CLI smoke on tiny synthetic inputs."""
import csv
import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from defectlab.common import hash_file, load_json, write_json
from defectlab.selection import select


def test_actual_resnet_cli_roundtrip(tmp_path):
    data = tmp_path / 'data'
    for role in ('train', 'test'):
        (data / role).mkdir(parents=True)
    for i in range(10):
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (40, 48, 3), dtype=np.uint8)).save(
            data / ('train' if i < 8 else 'test') / f'{i}.png')
    for name, header, rows in [
        ('train.csv', ['image_path', 'label'], [(f'{i}.png', i % 2) for i in range(8)]),
        ('sample_submission.csv', ['image_path', 'anomaly_score'], [('9.png', 0), ('8.png', 0)]),
    ]:
        with (data / name).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    def cli(*args):
        import os
        result = subprocess.run([sys.executable, '-m', 'defectlab', *map(str, args)],
                                capture_output=True, text=True,
                                env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    cli('verify', data)
    cli('profile', data, tmp_path / 'eda')
    split = tmp_path / 'split.json'
    cli('split', tmp_path / 'eda/profile.json', split, '--val-fraction', '.5')
    cfg = tmp_path / 'config.json'
    write_json(cfg, {'image_size': 32, 'batch_size': 2, 'epochs': 1, 'device': 'cpu', 'seed': 3})
    run = tmp_path / 'run'
    trained = cli('train', data, split, cfg, run)
    validation, reloaded = tmp_path / 'validation.json', tmp_path / 'reloaded.json'
    cli('evaluate', data, split, trained['best_checkpoint'], validation)
    cli('evaluate', data, split, trained['best_checkpoint'], reloaded)
    reproduction = cli('reproduce', validation, reloaded, '--probability-atol', '0', '--auroc-atol', '0')
    assert reproduction['passed']
    write_json(tmp_path / 'reproduction.json', reproduction)
    output = tmp_path / 'submission.csv'
    cli('predict', data, data / 'sample_submission.csv', trained['best_checkpoint'], output)
    assert [r['image_path'] for r in csv.DictReader(output.open())] == ['9.png', '8.png']
    cli('ensemble', tmp_path / 'ensemble.json', validation, reloaded, '--weights', '.5', '.5')
    cli('select', validation, str(output) + '.json', tmp_path / 'reproduction.json',
        tmp_path / 'selection.json', '--reason', 'Synthetic local functionality check')
    assert load_json(tmp_path / 'selection.json')['platform_submitted'] is False
    cli('results', tmp_path / 'report', run / 'run.json')
    assert (tmp_path / 'report/results.csv').exists()
    # Real optimizer-state resume and run ID continuity, not only checkpoint read.
    write_json(cfg, {'image_size': 32, 'batch_size': 2, 'epochs': 2, 'device': 'cpu', 'seed': 3})
    resumed = cli('train', data, split, cfg, run, '--resume', run / 'last.pt')
    assert resumed['completed_epochs'] == 2 and resumed['run_id'] == trained['run_id']
    torch.set_num_threads(1)
    state = torch.load(run / 'last.pt', weights_only=False)
    assert state['epoch'] == 2 and state['optimizer']['state']


def test_selection_rejects_wrong_reproduction(tmp_path):
    validation = {'kind': 'validation', 'dataset_hash': 'd', 'split_hash': 's'}
    test = {'kind': 'test', 'dataset_hash': 'd', 'split_hash': 's'}
    for name, value in [('v', validation), ('t', test), ('r', {'passed': True, 'reference_sha256': 'wrong'})]:
        write_json(tmp_path / name, value)
    with pytest.raises(ValueError, match='reference'):
        select(tmp_path / 'v', tmp_path / 't', tmp_path / 'r', tmp_path / 'out', 'reason')
    assert hash_file(tmp_path / 'v')


def test_selection_rejects_replaced_checkpoint(tmp_path):
    weights = tmp_path / 'weights.pt'
    weights.write_bytes(b'evaluated')
    provenance = {'dataset_hash': 'd', 'split_hash': 's', 'protocol_hash': 'p',
                  'checkpoint_sha256': hash_file(weights), 'checkpoint_path': str(weights)}
    write_json(tmp_path / 'v', {**provenance, 'kind': 'validation'})
    write_json(tmp_path / 't', {**provenance, 'kind': 'test'})
    write_json(tmp_path / 'r', {'passed': True, 'reference_sha256': hash_file(tmp_path / 'v')})
    weights.write_bytes(b'replaced')
    with pytest.raises(ValueError, match='checkpoint bytes'):
        select(tmp_path / 'v', tmp_path / 't', tmp_path / 'r', tmp_path / 'out', 'reason')
