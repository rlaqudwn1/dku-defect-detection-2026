"""End-to-end CLI smoke: verify -> profile -> folds -> train -> evaluate (TTA) -> errors -> predict -> ensemble."""
import csv
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from defectlab.common import write_json


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
        result = subprocess.run([sys.executable, '-m', 'defectlab', *map(str, args)],
                                capture_output=True, text=True,
                                env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    cli('verify', data)
    cli('profile', data, tmp_path / 'eda')
    folds = tmp_path / 'folds.json'
    cli('folds', tmp_path / 'eda/profile.json', folds, '--n-splits', '2', '--seed', '42')

    cfg = tmp_path / 'config.json'
    write_json(cfg, {
        'image_size': 32, 'batch_size': 2, 'epochs': 1, 'device': 'cpu', 'seed': 3,
        'augment': 'basic', 'scheduler': 'cosine', 'warmup_epochs': 0.5, 'grad_clip': 1.0,
    })
    run = tmp_path / 'run'
    trained = cli('train', data, folds, cfg, run, '--fold', '0', '--fast')
    assert trained['fold'] == 0

    validation, reloaded = tmp_path / 'validation.json', tmp_path / 'reloaded.json'
    cli('evaluate', data, folds, trained['best_checkpoint'], validation, '--fold', '0', '--fast')
    cli('evaluate', data, folds, trained['best_checkpoint'], reloaded, '--fold', '0', '--fast')
    reproduction = cli('reproduce', validation, reloaded, '--probability-atol', '0', '--auroc-atol', '0')
    assert reproduction['passed']

    # Error-driven EDA sheet generation
    err = cli('errors', data, validation, tmp_path / 'errors', '--top-k', '2')
    assert (tmp_path / 'errors/fp_top2.jpg').exists()
    assert (tmp_path / 'errors/fn_top2.jpg').exists()
    assert err['total_samples'] == 4

    # Test prediction with TTA
    output = tmp_path / 'submission.csv'
    cli('predict', data, data / 'sample_submission.csv', trained['best_checkpoint'], output, '--tta', '--fast')
    assert [r['image_path'] for r in csv.DictReader(output.open())] == ['9.png', '8.png']

    # Rank-averaged ensemble
    ens = cli('ensemble', tmp_path / 'ensemble.json', validation, reloaded, '--weights', '.5', '.5', '--method', 'rank')
    assert ens['recipe']['method'] == 'rank-weighted-mean'

    # Real optimizer-state resume and run ID continuity
    write_json(cfg, {
        'image_size': 32, 'batch_size': 2, 'epochs': 2, 'device': 'cpu', 'seed': 3,
        'augment': 'basic', 'scheduler': 'cosine', 'warmup_epochs': 0.5, 'grad_clip': 1.0,
    })
    resumed = cli('train', data, folds, cfg, run, '--fold', '0', '--resume', run / 'last.pt', '--fast')
    assert resumed['completed_epochs'] == 2 and resumed['run_id'] == trained['run_id']
    torch.set_num_threads(1)
    state = torch.load(run / 'last.pt', weights_only=False)
    assert state['epoch'] == 2 and state['optimizer']['state'] and state['scheduler'] is not None
