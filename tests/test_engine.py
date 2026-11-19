"""Bounded synthetic tests; no competition data or remote services are accessed."""
import csv
from pathlib import Path
import tempfile
import os
import json
import types
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image
import torch

from defectlab import engine
from defectlab.common import hash_file, load_json, write_json
from defectlab.data import dataset_hash, _rows


class EngineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        for role in ('train', 'test'):
            (self.data / role).mkdir()
        train_ids = [f'train_{i}.png' for i in range(8)]
        self.test_ids = ['test_1.png', 'test_0.png']
        for index, image_id in enumerate(train_ids + self.test_ids):
            role = 'train' if index < 8 else 'test'
            array = np.random.default_rng(index).integers(0, 255, (20+index, 40, 3), dtype=np.uint8)
            Image.fromarray(array).save(self.data / role / image_id)
        with (self.data / 'train.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_path', 'label'])
            writer.writerows((i, n % 2) for n, i in enumerate(train_ids))
        with (self.data / 'sample_submission.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_path', 'anomaly_score'])
            writer.writerows((i, 0) for i in self.test_ids)
        records, _ = _rows(self.data)
        for record in records:
            record['sha256'] = hash_file(self.data / record['relative_path'])
        self.split = {'schema_version': 1, 'dataset_hash': dataset_hash(records),
                      'train_ids': train_ids[:4], 'val_ids': train_ids[4:],
                      'labels': {i: n % 2 for n, i in enumerate(train_ids)},
                      'groups': {i: i for i in train_ids}, 'near_duplicate_status': 'unassessed'}
        self.split_path = self.root / 'split.json'
        write_json(self.split_path, self.split)
        self.config = {'model': 'resnet18', 'image_size': 32, 'batch_size': 2, 'epochs': 2,
                       'lr': .01, 'seed': 32, 'device': 'cpu', 'tracking': {'mode': 'disabled'}}
        self.config_path = self.root / 'config.json'
        write_json(self.config_path, self.config)

    @staticmethod
    def tiny(cfg, initialize=False):
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3*32*32, 1))

    def test_preprocess_and_real_resnet_forward(self):
        value = engine.preprocess(self.data / 'train/train_0.png', 32)
        self.assertEqual(tuple(value.shape), (3, 32, 32))
        self.assertTrue(torch.isfinite(value).all())
        model = engine._model(self.config).eval()
        with torch.inference_mode():
            self.assertEqual(tuple(model(value[None]).shape), (1, 1))
        self.assertEqual(engine._spec(self.config)['positive_label'], 1)

    def test_train_resume_predict_ensemble_reproduce(self):
        with patch.object(engine, '_model', self.tiny):
            full = engine.train(self.data, self.split_path, self.config_path, self.root / 'full')
            write_json(self.config_path, {**self.config, 'epochs': 1})
            engine.train(self.data, self.split_path, self.config_path, self.root / 'resumed')
            write_json(self.config_path, self.config)
            resumed = engine.train(self.data, self.split_path, self.config_path, self.root / 'resumed',
                                   resume=self.root / 'resumed/last.pt')
            a = torch.load(full['last_checkpoint'], weights_only=False)
            b = torch.load(resumed['last_checkpoint'], weights_only=False)
            self.assertEqual(a['epoch'], b['epoch'])
            for name in a['model']:
                self.assertTrue(torch.equal(a['model'][name], b['model'][name]), name)
            self.assertEqual(a['optimizer']['param_groups'], b['optimizer']['param_groups'])
            before = {p.name: hash_file(p) for p in (self.root / 'resumed').iterdir() if p.is_file()}
            write_json(self.config_path, {**self.config, 'epochs': 3})
            with self.assertRaisesRegex(ValueError, 'different run'):
                engine.train(self.data, self.split_path, self.config_path, self.root / 'resumed',
                             resume=self.root / 'full/last.pt')
            after = {p.name: hash_file(p) for p in (self.root / 'resumed').iterdir() if p.is_file()}
            self.assertEqual(before, after, 'rejected cross-run resume must preserve every output byte')
            write_json(self.config_path, self.config)
            reference = self.root / 'reference.json'
            engine.evaluate(self.data, self.split_path, full['best_checkpoint'], reference)
            current = self.root / 'current.json'
            other = engine.evaluate(self.data, self.split_path, full['best_checkpoint'], current)
            other['rows'].reverse()
            write_json(current, other)
            result = engine.reproduce(reference, current, 0, 0)
            self.assertTrue(result['passed'])
            combined = engine.ensemble([reference, current], [.25, .75], self.root / 'ensemble.json')
            self.assertEqual(combined['auroc'], other['auroc'])
            relocated = self.root / 'relocated' / 'evaluation.json'
            write_json(relocated, other)
            repeated = engine.ensemble([relocated, reference], [.25, .75], self.root / 'rerun.json')
            self.assertEqual(combined['recipe_hash'], repeated['recipe_hash'])
            self.assertTrue(engine.reproduce(self.root / 'ensemble.json', self.root / 'rerun.json', 0, 0)['passed'])
            self.assertNotEqual(combined['recipe']['members'][0]['prediction_path'],
                                repeated['recipe']['members'][0]['prediction_path'])
            other['rows'][0]['label'] = 1 - other['rows'][0]['label']
            write_json(current, other)
            with self.assertRaisesRegex(ValueError, 'label disagreement'):
                engine.ensemble([reference, current], [.5, .5], self.root / 'bad.json')
            prediction = engine.predict(self.data, self.data / 'sample_submission.csv',
                                        full['best_checkpoint'], self.root / 'test.csv')
            self.assertEqual([r['image_path'] for r in prediction['rows']], self.test_ids)
            with self.assertRaisesRegex(ValueError, 'requires a frozen validation recipe'):
                engine.ensemble([self.root / 'test.csv', self.root / 'test.csv'], [.25, .75], self.root / 'test-ensemble.json')
            final = engine.ensemble([self.root / 'test.csv', self.root / 'test.csv'], [.25, .75],
                                    self.root / 'test-ensemble.json',
                                    frozen_recipe_path=self.root / 'ensemble.json.recipe.json')
            self.assertEqual(final['recipe_hash'], combined['recipe_hash'])
            with self.assertRaisesRegex(ValueError, 'differ from frozen'):
                engine.ensemble([self.root / 'test.csv', self.root / 'test.csv'], [.5, .5],
                                self.root / 'changed-test-ensemble.json',
                                frozen_recipe_path=self.root / 'ensemble.json.recipe.json')
            self.assertTrue(engine.reproduce(self.root / 'test.csv', self.root / 'test.csv', 0, 0)['passed'])
            self.assertTrue((self.root / 'full/model-manifest.json').exists())
            self.assertGreater(len((self.root / 'full/events.jsonl').read_text().splitlines()), 2)

    def test_leakage_provenance_and_mutation(self):
        leaking = {**self.split, 'val_ids': self.split['train_ids']}
        write_json(self.split_path, leaking)
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            engine.train(self.data, self.split_path, self.config_path, self.root / 'bad')
        write_json(self.split_path, self.split)
        image = self.data / 'train/train_0.png'
        image.write_bytes(image.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'dataset content'):
            engine.train(self.data, self.split_path, self.config_path, self.root / 'bad')

    def test_pretrained_never_downloaded_and_requires_provenance(self):
        with self.assertRaisesRegex(ValueError, 'provenance missing'):
            engine._model({**self.config, 'pretrained': {'path': 'missing.pt'}}, initialize=True)
        # Loading a trained checkpoint does not re-open initialization files.
        with patch.object(engine, '_model', self.tiny):
            result = engine.train(self.data, self.split_path, self.config_path, self.root / 'run')
            checkpoint = torch.load(result['best_checkpoint'], weights_only=False)
            checkpoint['config']['pretrained'] = {'path': 'now-missing.pt'}
            torch.save(checkpoint, self.root / 'missing-init.pt')
            engine._load_model(self.root / 'missing-init.pt')

    def test_ensemble_rejects_forged_holdout_and_reproduction_detects_drift(self):
        with patch.object(engine, '_model', self.tiny):
            run = engine.train(self.data, self.split_path, self.config_path, self.root / 'run')
            reference = self.root / 'reference.json'
            prediction = engine.evaluate(self.data, self.split_path, run['best_checkpoint'], reference)
        altered = self.root / 'altered.json'
        changed = {**prediction, 'rows': [dict(r) for r in prediction['rows']]}
        changed['rows'][0]['anomaly_score'] = 1. - changed['rows'][0]['anomaly_score']
        write_json(altered, changed)
        self.assertFalse(engine.reproduce(reference, altered, 0, 0)['passed'])
        checkpoint = torch.load(run['best_checkpoint'], weights_only=False)
        checkpoint['train_ids'] = checkpoint['train_ids'] + [checkpoint['val_ids'][0]]
        torch.save(checkpoint, self.root / 'leaked.pt')
        forged = {**prediction, 'train_ids': checkpoint['train_ids'],
                  'checkpoint_path': str(self.root / 'leaked.pt'),
                  'checkpoint_sha256': hash_file(self.root / 'leaked.pt')}
        write_json(altered, forged)
        with self.assertRaisesRegex(ValueError, 'training leakage'):
            engine.ensemble([reference, altered], [.5, .5], self.root / 'bad.json')
        forged['checkpoint_sha256'] = '0' * 64
        write_json(altered, forged)
        with self.assertRaisesRegex(ValueError, 'checkpoint hash mismatch'):
            engine.ensemble([reference, altered], [.5, .5], self.root / 'bad.json')
        self.assertEqual(engine._auroc([{'label': 0, 'anomaly_score': .1},
                                      {'label': 1, 'anomaly_score': .9}]), 1.)

    def test_training_budget_failure_is_preserved(self):
        write_json(self.config_path, {**self.config, 'max_runtime_seconds': 1e-9})
        with patch.object(engine, '_model', self.tiny):
            with self.assertRaisesRegex(TimeoutError, 'budget exhausted'):
                engine.train(self.data, self.split_path, self.config_path, self.root / 'timeout')
        self.assertEqual(load_json(self.root / 'timeout/run.json')['status'], 'failed')
        self.assertIn('TimeoutError', load_json(self.root / 'timeout/run.json')['error'])

    def test_tune_records_failure_and_candidate_bound(self):
        plan = self.root / 'plan.json'
        write_json(plan, {'max_candidates': 2, 'time_budget_seconds': 60,
                          'candidates': [{'config': {**self.config, 'model': 'unsupported'}},
                                         {'config': self.config}, {'config': self.config}]})
        with patch.object(engine, '_model', self.tiny):
            result = engine.tune(self.data, self.split_path, plan, self.root / 'tune')
        self.assertEqual(result['attempted_candidates'], 2)
        self.assertEqual([r['status'] for r in result['results']], ['failed', 'completed'])
        self.assertEqual(result['selection_attempts'], 1)
        self.assertEqual(result['best_candidate'], 1)

    def test_multimodel_forward_freeze_pos_weight_and_vit_pos_interp(self):
        value = engine.preprocess(self.data / 'train/train_0.png', 32)
        dummy_weights = self.root / 'dummy.pth'
        dummy_weights.write_bytes(b'synthetic-provenance-placeholder')
        provenance = {'path': str(dummy_weights), 'source_url': 'https://example.com/w.pth',
                      'version': 'test', 'publication_date': '2026-11-15',
                      'sha256': hash_file(dummy_weights), 'license': 'BSD-3-Clause'}
        for name in ('resnet18', 'convnext_tiny', 'swin_t', 'vit_b_16'):
            cfg_path = self.root / f'{name}.json'
            write_json(cfg_path, {**self.config, 'model': name, 'freeze_backbone': True,
                                  'pos_weight': 4.0, 'pretrained': provenance})
            cfg = engine._config(cfg_path)
            self.assertEqual(engine._spec(cfg)['architecture'], f'torchvision.{name}')
            self.assertTrue(engine._spec(cfg)['freeze_backbone'])
            model = engine._model(cfg).eval()
            with torch.inference_mode():
                self.assertEqual(tuple(model(value[None]).shape), (1, 1))
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            self.assertEqual(len(trainable), 2)

        for bad_cfg, pattern in (
            ({**self.config, 'model': 'vit_b_16', 'image_size': 36}, 'divisible by 16'),
            ({**self.config, 'freeze_backbone': True}, 'requires pretrained'),
            ({**self.config, 'freeze_backbone': 'yes'}, 'boolean'),
            ({**self.config, 'pos_weight': 0}, 'pos_weight'),
            ({**self.config, 'pos_weight': -1.0}, 'pos_weight'),
            ({**self.config, 'pos_weight': True}, 'pos_weight'),
        ):
            bad_path = self.root / 'bad-cfg.json'
            write_json(bad_path, bad_cfg)
            with self.assertRaisesRegex(ValueError, pattern):
                engine._config(bad_path)

        # Real ResNet18 frozen training & resume: backbone weights and BatchNorm running stats stay frozen.
        from torchvision.models import resnet18, vit_b_16
        res_init_state = resnet18(weights=None).state_dict()
        res_cfg_path = self.root / 'real-res-freeze.json'
        write_json(res_cfg_path, {**self.config, 'model': 'resnet18', 'epochs': 1,
                                  'freeze_backbone': True, 'pretrained': provenance})
        with patch.object(engine.torch, 'load', return_value=res_init_state):
            res_freeze_run = engine.train(self.data, self.split_path, res_cfg_path, self.root / 'run-real-freeze')
        ckpt_1 = torch.load(res_freeze_run['last_checkpoint'], weights_only=False)
        self.assertTrue(torch.equal(ckpt_1['model']['conv1.weight'], res_init_state['conv1.weight']))
        self.assertTrue(torch.equal(ckpt_1['model']['bn1.running_mean'], res_init_state['bn1.running_mean']))
        self.assertTrue(torch.equal(ckpt_1['model']['bn1.running_var'], res_init_state['bn1.running_var']))
        write_json(res_cfg_path, {**self.config, 'model': 'resnet18', 'epochs': 2,
                                  'freeze_backbone': True, 'pretrained': provenance})
        res_resumed = engine.train(self.data, self.split_path, res_cfg_path, self.root / 'run-real-freeze',
                                   resume=res_freeze_run['last_checkpoint'])
        ckpt_2 = torch.load(res_resumed['last_checkpoint'], weights_only=False)
        self.assertEqual(ckpt_2['epoch'], 2)
        self.assertTrue(torch.equal(ckpt_2['model']['bn1.running_mean'], res_init_state['bn1.running_mean']))
        self.assertFalse(torch.equal(ckpt_2['model']['fc.weight'], ckpt_1['model']['fc.weight']))

        # Verify pos_weight alters gradients/weights and cross-model ensembling works across architectures.
        def tiny_by_arch(cfg, initialize=False):
            backbone = torch.nn.Conv2d(3, 4, 3, padding=1)
            head = torch.nn.Linear(4, 1)
            mod = torch.nn.Sequential(backbone, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), head)
            if cfg.get('freeze_backbone'):
                for p in backbone.parameters():
                    p.requires_grad = False
            return mod

        with patch.object(engine, '_model', tiny_by_arch):
            cfg_plain = self.root / 'cfg-plain.json'
            cfg_res = self.root / 'cfg-res.json'
            cfg_conv = self.root / 'cfg-conv.json'
            write_json(cfg_plain, {**self.config, 'model': 'resnet18', 'epochs': 1,
                                   'freeze_backbone': True, 'pretrained': provenance})
            write_json(cfg_res, {**self.config, 'model': 'resnet18', 'epochs': 1,
                                 'freeze_backbone': True, 'pos_weight': 10.0, 'pretrained': provenance})
            write_json(cfg_conv, {**self.config, 'model': 'convnext_tiny', 'epochs': 1,
                                  'freeze_backbone': True, 'pretrained': provenance})
            plain_run = engine.train(self.data, self.split_path, cfg_plain, self.root / 'run-plain')
            res_run = engine.train(self.data, self.split_path, cfg_res, self.root / 'run-res')
            conv_run = engine.train(self.data, self.split_path, cfg_conv, self.root / 'run-conv')
            ckpt_plain = torch.load(plain_run['best_checkpoint'], weights_only=False)
            ckpt = torch.load(res_run['best_checkpoint'], weights_only=False)
            self.assertFalse(torch.equal(ckpt['model']['3.weight'], ckpt_plain['model']['3.weight']))
            self.assertNotAlmostEqual(res_run['history'][0]['loss'], plain_run['history'][0]['loss'])
            engine._seed(self.config['seed'])
            fresh = tiny_by_arch(engine._config(cfg_res))
            self.assertTrue(torch.equal(ckpt['model']['0.weight'], fresh[0].weight))
            self.assertFalse(torch.equal(ckpt['model']['3.weight'], fresh[3].weight))
            res_eval = self.root / 'eval-res.json'
            conv_eval = self.root / 'eval-conv.json'
            engine.evaluate(self.data, self.split_path, res_run['best_checkpoint'], res_eval)
            engine.evaluate(self.data, self.split_path, conv_run['best_checkpoint'], conv_eval)
            ens = engine.ensemble([res_eval, conv_eval], [0.5, 0.5], self.root / 'ens-cross.json')
            self.assertIn('auroc', ens)

        # Synthetic 224-grid ViT pretrained state_dict interpolated to image_size=32 locally (no download).
        base_state = vit_b_16(weights=None, image_size=32).state_dict()
        pos_224 = torch.randn(1, 1 + 14 * 14, 768)
        expected_cls = pos_224[:, :1, :].clone()
        state_224 = {**base_state, 'encoder.pos_embedding': pos_224}
        vit_weights = self.root / 'vit224.pth'
        vit_weights.write_bytes(b'synthetic-vit-provenance-placeholder')
        vit_cfg = {**self.config, 'model': 'vit_b_16', 'image_size': 32,
                   'pretrained': {'path': str(vit_weights), 'source_url': 'https://example.com/vit.pth',
                                  'version': 'test', 'publication_date': '2026-11-15',
                                  'sha256': hash_file(vit_weights), 'license': 'BSD-3-Clause'}}
        with patch.object(engine.torch, 'load', return_value=state_224):
            loaded_vit = engine._model(vit_cfg, initialize=True)
        interpolated = loaded_vit.state_dict()['encoder.pos_embedding']
        self.assertEqual(tuple(interpolated.shape), (1, 5, 768))
        self.assertTrue(torch.equal(interpolated[:, :1, :], expected_cls))

    def test_seed_cublas_env_and_amp_overflow_handling(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
            engine._seed(42)
            self.assertEqual(os.environ.get('CUBLAS_WORKSPACE_CONFIG'), ':4096:8')
            self.assertEqual(engine._environment('cpu')['cublas_workspace_config'], ':4096:8')

        param = torch.nn.Parameter(torch.ones(2))
        param.grad = torch.tensor([float('inf'), 1.0])
        opt = torch.optim.SGD([param], lr=0.1)
        disabled_scaler = torch.amp.GradScaler('cuda', enabled=False)
        with self.assertRaisesRegex(ValueError, 'nonfinite gradient'):
            engine._step_optimizer(disabled_scaler, opt, [param], amp=False, consecutive_overflows=0)

        class _BackoffScaler:
            def __init__(self, scales):
                self.scales = list(scales)
                self.idx = 0

            def unscale_(self, optimizer):
                pass

            def get_scale(self):
                return self.scales[self.idx]

            def step(self, optimizer):
                pass

            def update(self):
                self.idx += 1

        transient = _BackoffScaler([65536.0, 32768.0, 16384.0, 16384.0])
        skipped, streak = engine._step_optimizer(transient, opt, [param], amp=True, consecutive_overflows=0)
        self.assertEqual((skipped, streak), (True, 1))
        skipped, streak = engine._step_optimizer(transient, opt, [param], amp=True, consecutive_overflows=streak)
        self.assertEqual((skipped, streak), (True, 2))
        skipped, streak = engine._step_optimizer(transient, opt, [param], amp=True, consecutive_overflows=streak)
        self.assertEqual((skipped, streak), (False, 0))

        fatal = _BackoffScaler([65536.0 / (2 ** i) for i in range(12)])
        streak = 0
        for _ in range(9):
            skipped, streak = engine._step_optimizer(fatal, opt, [param], amp=True, consecutive_overflows=streak)
            self.assertTrue(skipped)
        with self.assertRaisesRegex(ValueError, 'AMP gradient overflow persisted'):
            engine._step_optimizer(fatal, opt, [param], amp=True, consecutive_overflows=streak)


class TrackingTests(unittest.TestCase):
    def test_real_offline_sdk_writes_metrics_artifact_and_finish_without_credentials(self):
        from defectlab.tracking import Tracker
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            weight = output / 'weights.bin'
            weight.write_bytes(b'synthetic offline artifact bytes')
            environment = {'WANDB_MODE': 'offline', 'WANDB_API_KEY': '', 'WANDB_SILENT': 'true',
                           'WANDB_CONSOLE': 'off'}
            with patch.dict(os.environ, environment):
                tracker = Tracker(output, {'mode': 'offline', 'project': 'defectlab-offline-test'},
                                  'synthetic-offline-test', {'seed': 42})
                tracker.log({'loss': .125, 'val_auroc': .75}, step=1)
                receipt = tracker.artifact(weight, 'synthetic-weight')
                run_dir = Path(tracker.run.dir)
                tracker.finish()
            self.assertTrue(run_dir.exists())
            self.assertTrue(list((output / 'wandb').glob('offline-run-*/run-*.wandb')))
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e.get('val_auroc') == .75 for e in events))
            self.assertEqual(events[-1]['status'], 'completed')
            self.assertEqual(receipt['sha256'], hash_file(weight))
            self.assertIsNone(receipt['cloud_version'])
            self.assertFalse(receipt['upload_acknowledged'])

    def test_mocked_online_adapter_records_acknowledged_exact_artifact_version(self):
        from defectlab.tracking import Tracker
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            weight = output / 'weights.bin'
            weight.write_bytes(b'synthetic mocked-online artifact bytes')
            run = MagicMock()
            acknowledged = types.SimpleNamespace(version='v7', qualified_name='team/project/model:v7')
            run.log_artifact.return_value.wait.return_value = acknowledged
            sdk = types.SimpleNamespace(init=MagicMock(return_value=run),
                                        Settings=MagicMock(), Artifact=MagicMock())
            with patch.dict(sys.modules, {'wandb': sdk}):
                tracker = Tracker(output, {'mode': 'online', 'entity': 'team', 'project': 'project'},
                                  'run-fixed', {'seed': 42})
                tracker.log({'loss': .25}, step=3)
                receipt = tracker.artifact(weight, 'model')
                tracker.finish('failed')
            kwargs = sdk.init.call_args.kwargs
            self.assertEqual((kwargs['entity'], kwargs['project'], kwargs['id'], kwargs['mode']),
                             ('team', 'project', 'run-fixed', 'online'))
            self.assertEqual(kwargs['config'], {'seed': 42})
            run.log.assert_any_call({'loss': .25}, step=3)
            sdk.Artifact.assert_called_once_with(name='model', type='model', metadata={'sha256': hash_file(weight)})
            sdk.Artifact.return_value.add_file.assert_called_once_with(str(weight))
            run.log_artifact.return_value.wait.assert_called_once_with(timeout=60)
            self.assertEqual(receipt['cloud_qualified_name'], 'team/project/model:v7')
            self.assertEqual(receipt['cloud_version'], 'v7')
            self.assertTrue(receipt['upload_acknowledged'])
            run.finish.assert_called_once_with(exit_code=1)


if __name__ == '__main__':
    unittest.main()
