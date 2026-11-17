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
