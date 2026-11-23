"""Tests for the upgraded Senior ML Engineer loop harness (folds, errors, aug, scheduler, TTA, rank/logit ensemble)."""
import csv
import sys
import types
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

from defectlab import data, engine, errors
from defectlab.common import write_json


def _make_dataset(root, n_train=20, n_test=4):
    for role in ('train', 'test'):
        (root / role).mkdir(parents=True, exist_ok=True)
    for i in range(n_train):
        arr = np.random.default_rng(i).integers(0, 255, (36, 48, 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / 'train' / f'tr_{i:02d}.png')
    for i in range(n_test):
        arr = np.random.default_rng(100 + i).integers(0, 255, (36, 48, 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / 'test' / f'te_{i:02d}.png')
    with (root / 'train.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_path', 'label'])
        for i in range(n_train):
            w.writerow([f'tr_{i:02d}.png', i % 2])
    with (root / 'sample_submission.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_path', 'anomaly_score'])
        for i in range(n_test):
            w.writerow([f'te_{i:02d}.png', 0])


def test_make_folds_stratified_and_group_safe(tmp_path):
    ds = tmp_path / 'ds'
    _make_dataset(ds, n_train=20)
    prof = tmp_path / 'eda'
    data.profile(ds, prof)
    # Put tr_00 and tr_02 (both label 0) in the same group
    groups_path = tmp_path / 'groups.json'
    write_json(groups_path, {'tr_00.png': 'grpA', 'tr_02.png': 'grpA'})
    folds_path = tmp_path / 'folds.json'
    out = data.make_folds(prof / 'profile.json', folds_path, n_splits=5, seed=42, groups_path=groups_path)
    assert out['n_splits'] == 5
    all_val = []
    for k in range(5):
        f = out['folds'][str(k)]
        assert set(f['train_ids']).isdisjoint(f['val_ids'])
        assert len(f['train_ids']) + len(f['val_ids']) == 20
        # grpA must never be split across train and val
        in_val = {'tr_00.png', 'tr_02.png'} & set(f['val_ids'])
        assert len(in_val) in (0, 2)
        all_val.extend(f['val_ids'])
    assert len(all_val) == 20 and len(set(all_val)) == 20


def test_augment_resize_scheduler_tta_and_rank_logit_ensemble(tmp_path):
    torch.set_num_threads(1)
    ds = tmp_path / 'ds'
    _make_dataset(ds, n_train=12)
    prof = tmp_path / 'eda'
    data.profile(ds, prof)
    folds_path = tmp_path / 'folds.json'
    data.make_folds(prof / 'profile.json', folds_path, n_splits=3, seed=7)

    # Stretch + strong augment produces valid finite tensor
    t = engine.preprocess(ds / 'train/tr_00.png', 32, resize_mode='stretch', augment='strong')
    assert tuple(t.shape) == (3, 32, 32) and torch.isfinite(t).all()

    def tiny(cfg, initialize=False):
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 1))

    cfg_path = tmp_path / 'cfg.json'
    write_json(cfg_path, {
        'model': 'resnet18', 'image_size': 32, 'resize_mode': 'stretch',
        'augment': 'strong', 'batch_size': 4, 'epochs': 2, 'lr': 0.01,
        'scheduler': 'cosine', 'warmup_epochs': 0.5, 'grad_clip': 1.0,
        'label_smoothing': 0.05, 'num_workers': 0, 'device': 'cpu', 'fast_validate': True,
    })

    with patch.object(engine, '_model', tiny):
        run = engine.train(ds, folds_path, cfg_path, tmp_path / 'run0', fold=0)
        assert run['fold'] == 0 and len(run['history']) == 2
        ev1 = engine.evaluate(ds, folds_path, run['best_checkpoint'], tmp_path / 'ev1.json',
                              device='cpu', tta=False, fold=0, fast_validate=True)
        assert ev1['tta'] is False
        ev2 = engine.evaluate(ds, folds_path, run['best_checkpoint'], tmp_path / 'ev2.json',
                              device='cpu', tta=True, fold=0, fast_validate=True)
        assert ev2['tta'] is True
        ens_rank = engine.ensemble([tmp_path / 'ev1.json', tmp_path / 'ev2.json'], [0.5, 0.5],
                                   tmp_path / 'ens_rank.json', method='rank')
        assert ens_rank['recipe']['method'] == 'rank-weighted-mean'
        ens_logit = engine.ensemble([tmp_path / 'ev1.json', tmp_path / 'ev2.json'], [0.5, 0.5],
                                    tmp_path / 'ens_logit.json', method='logit')
        assert ens_logit['recipe']['method'] == 'logit-weighted-mean'
        assert all(0.0 <= r['anomaly_score'] <= 1.0 for r in ens_logit['rows'])

    # Error-driven EDA sheet
    err = errors.analyze_errors(ds, tmp_path / 'ev1.json', tmp_path / 'err', top_k=2)
    assert (tmp_path / 'err/fp_top2.jpg').exists()
    assert (tmp_path / 'err/fn_top2.jpg').exists()
    assert len(err['false_positives']) == 2 and len(err['false_negatives']) == 2
    assert err['false_positives'][0]['anomaly_score'] >= err['false_positives'][1]['anomaly_score']
    assert err['false_negatives'][0]['anomaly_score'] <= err['false_negatives'][1]['anomaly_score']


def test_timm_lazy_model_dispatch():
    created = {}

    def fake_create_model(name, pretrained=False, num_classes=1):
        created['args'] = (name, pretrained, num_classes)
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, num_classes))

    fake_timm = types.SimpleNamespace(create_model=fake_create_model)
    with patch.dict(sys.modules, {'timm': fake_timm}):
        cfg = {'model': 'timm:convnext_small.fb_in22k_ft_in1k_384', 'pretrained_timm': True, 'image_size': 32}
        model = engine._model(cfg, initialize=True)
        assert created['args'] == ('convnext_small.fb_in22k_ft_in1k_384', True, 1)
        assert model is not None


def test_num_workers_positive_allowed_in_config(tmp_path):
    cfg_path = tmp_path / 'c.json'
    write_json(cfg_path, {'num_workers': 4, 'device': 'cpu'})
    cfg = engine._config(cfg_path)
    assert cfg['num_workers'] == 4
    write_json(cfg_path, {'num_workers': -1, 'device': 'cpu'})
    with pytest.raises(ValueError, match='num_workers'):
        engine._config(cfg_path)


def test_brief_scans_state_experiments_runs_without_torch(tmp_path):
    from defectlab.brief import build_brief
    (tmp_path / 'AGENTS.md').write_text('# Rules\n', encoding='utf-8')
    (tmp_path / 'STATE.md').write_text('# State\n- ok\n', encoding='utf-8')
    (tmp_path / 'EXPERIMENTS.md').write_text(
        '| Exp | Score |\n|---|---|\n| E01 | 0.95 |\n', encoding='utf-8'
    )
    run_dir = tmp_path / 'runs' / 'exp01'
    run_dir.mkdir(parents=True)
    write_json(run_dir / 'run.json', {'status': 'completed', 'run_id': 'abc', 'fold': 0,
                                      'best_auroc': 0.9512, 'completed_epochs': 10})
    err_dir = tmp_path / 'runs' / 'exp01' / 'errors'
    err_dir.mkdir(parents=True)
    write_json(err_dir / 'errors.json', {'auroc': 0.9512, 'top_k': 20, 'total_samples': 200})

    summary = build_brief(tmp_path, 'runs', top_n=3)
    assert summary['state_bloat_warning'] is False
    assert summary['experiments']['total_rows'] == 1
    assert summary['top_completed_runs'][0]['best_auroc'] == 0.9512
    assert summary['recent_error_reports'][0]['top_k'] == 20
