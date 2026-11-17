"""Thin, deterministic ResNet18 baseline and provenance-checked prediction tools.

Checkpoints are trusted local PyTorch files (pickle); never load untrusted files.
"""
import csv
import math
import platform
import inspect
import subprocess
import random
import time
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from .common import hash_file, hash_json, load_config, load_json, safe_path, write_json
from .tracking import Tracker


def _config(path):
    cfg = dict(load_config(path))
    defaults = dict(model='resnet18', image_size=224, batch_size=8, epochs=1,
                    lr=0.001, weight_decay=0.0, seed=42, device='cpu', amp=False,
                    num_workers=0, tracking={'mode': 'disabled'})
    cfg = {**defaults, **cfg}
    if cfg['model'] != 'resnet18':
        raise ValueError('only resnet18 is supported')
    for key in ('image_size', 'batch_size', 'epochs'):
        if type(cfg[key]) is not int or cfg[key] < (32 if key == 'image_size' else 1):
            raise ValueError(f'invalid {key}')
    if cfg['device'] not in ('cpu', 'cuda') or cfg['num_workers'] != 0:
        raise ValueError('device must be cpu/cuda; deterministic baseline requires num_workers=0')
    if cfg['amp'] and cfg['device'] != 'cuda':
        raise ValueError('AMP is supported only on CUDA')
    if cfg['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    for key in ('lr', 'weight_decay'):
        if not math.isfinite(cfg[key]) or cfg[key] < 0 or (key == 'lr' and cfg[key] == 0):
            raise ValueError(f'invalid {key}')
    if cfg.get('max_runtime_seconds') is not None and (not math.isfinite(cfg['max_runtime_seconds']) or cfg['max_runtime_seconds'] <= 0):
        raise ValueError('max_runtime_seconds must be finite and positive')
    return cfg


def _split(path):
    split = load_json(path)
    for key in ('dataset_hash', 'train_ids', 'val_ids', 'labels', 'groups'):
        if key not in split:
            raise ValueError(f'missing split {key}')
    train_ids, val_ids = split['train_ids'], split['val_ids']
    if (not train_ids or not val_ids or len(set(train_ids)) != len(train_ids)
            or len(set(val_ids)) != len(val_ids) or set(train_ids) & set(val_ids)):
        raise ValueError('invalid or overlapping split IDs')
    for ids in (train_ids, val_ids):
        if any(split['labels'].get(i) not in (0, 1) or i not in split['groups'] for i in ids):
            raise ValueError('split labels/groups incomplete')
        if {split['labels'][i] for i in ids} != {0, 1}:
            raise ValueError('both classes required in train and validation')
    if ({split['groups'][i] for i in train_ids} & {split['groups'][i] for i in val_ids}):
        raise ValueError('group leakage')
    return split


def _validate_dataset(root, expected_hash, split=None):
    from .data import _rows, dataset_hash
    records, _ = _rows(Path(root))
    for record in records:
        record['sha256'] = hash_file(safe_path(root, record['relative_path']))
    if split is not None:
        actual_labels = {r['image_path']: r['label'] for r in records if r['role'] == 'train'}
        selected = set(split['train_ids']) | set(split['val_ids'])
        if selected != set(actual_labels) or any(split['labels'][i] != actual_labels[i] for i in selected):
            raise ValueError('split IDs/labels differ from actual training CSV')
    if dataset_hash(records) != expected_hash:
        raise ValueError('actual dataset content/labels differ from recorded dataset hash')


def _output_guard(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError('outputs must be outside read-only dataset root')


def _image_path(root, image_id, role):
    path = safe_path(root, image_id)
    return path if Path(path).is_file() else safe_path(root, f'{role}/{image_id}')


def preprocess(path, image_size):
    """EXIF transpose, RGB, bilinear aspect-preserving resize/center pad, ImageNet norm."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        width, height = image.size
        scale = image_size / max(width, height)
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                             Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (image_size, image_size), (0, 0, 0))
        canvas.paste(image, ((image_size-image.width)//2, (image_size-image.height)//2))
        array = np.asarray(canvas, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


class _Images(Dataset):
    def __init__(self, root, ids, size, labels=None, role='train'):
        self.root, self.ids, self.size, self.labels, self.role = root, ids, size, labels, role

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        tensor = preprocess(_image_path(self.root, image_id, self.role), self.size)
        return tensor, float(self.labels[image_id]) if self.labels is not None else 0., image_id


def _spec(cfg):
    implementation = hash_json({fn.__name__: inspect.getsource(fn)
                                for fn in (preprocess, _model, _scores, _image_path)})
    return {'inference_implementation_sha256': implementation,'architecture': 'torchvision.resnet18', 'head': 'linear-1-logit',
            'positive_label': 1, 'probability': 'sigmoid-once',
            'preprocessing': {'image_size': cfg['image_size'], 'resize': 'bilinear-aspect-preserving',
                              'pad': 'center-black', 'color': 'RGB', 'exif_transpose': True,
                              'mean': [.485, .456, .406], 'std': [.229, .224, .225]}}


def _model(cfg, initialize=False):
    model = resnet18(weights=None)
    if initialize and cfg.get('pretrained'):
        source = cfg['pretrained']
        for field in ('path', 'source_url', 'version', 'publication_date', 'sha256', 'license'):
            if not source.get(field):
                raise ValueError(f'pretrained provenance missing {field}')
        if not source['source_url'].startswith('https://') or date.fromisoformat(source['publication_date']) > date(2026, 11, 15):
            raise ValueError('invalid pretrained source or publication cutoff')
        if hash_file(source['path']) != source['sha256']:
            raise ValueError('pretrained hash mismatch')
        model.load_state_dict(torch.load(source['path'], map_location='cpu', weights_only=True))
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _code_identity():
    package = Path(__file__).resolve().parent
    sources = {f'src/defectlab/{path.relative_to(package).as_posix()}': hash_file(path)
               for path in sorted(package.rglob('*.py'))}
    commit, dirty = None, None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=package,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'],
                                             cwd=package, stderr=subprocess.DEVNULL))
    except (OSError, subprocess.CalledProcessError):
        pass
    return {'git_commit': commit, 'git_dirty': dirty, 'source_files': sources,
            'source_tree_sha256': hash_json(sources)}


def _environment(device=None):
    import torchvision
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'cuda': torch.version.cuda, 'execution_device': device,
            'visible_gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def _scores(model, loader, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, labels, ids in loader:
            probabilities = model(images.to(device)).flatten().sigmoid().float().cpu().tolist()
            for image_id, label, probability in zip(ids, labels.tolist(), probabilities):
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError('invalid prediction')
                rows.append({'image_path': image_id, 'label': int(label), 'anomaly_score': probability})
    return rows


def _loader(root, ids, cfg, labels=None, role='train', shuffle=False):
    # BatchNorm needs >1 sample at the last feature map for batch size 1, size 32.
    return DataLoader(_Images(root, ids, cfg['image_size'], labels, role),
                      batch_size=cfg['batch_size'], shuffle=shuffle, num_workers=0)


def _auroc(rows):
    return float(roc_auc_score([r['label'] for r in rows], [r['anomaly_score'] for r in rows]))


def _save_checkpoint(path, state):
    temporary = Path(str(path) + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def train(data_root, split_path, config_path, output_dir, resume=None):
    cfg, split = _config(config_path), _split(split_path)
    _output_guard(data_root, output_dir)
    _validate_dataset(data_root, split['dataset_hash'], split)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any((output / name).exists() for name in ('run.json', 'best.pt', 'last.pt')):
        raise ValueError('output already contains a run; resume or choose a new output directory')
    checkpoint = None
    if resume is not None:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        if (output / 'run.json').exists() and load_json(output / 'run.json').get('run_id') != checkpoint['run_id']:
            raise ValueError('resume output belongs to a different run')
        for name in ('best.pt', 'last.pt'):
            if (output / name).exists():
                existing = torch.load(output / name, map_location='cpu', weights_only=False)
                if (existing['run_id'] != checkpoint['run_id'] or existing['split_hash'] != checkpoint['split_hash']
                        or existing['dataset_hash'] != checkpoint['dataset_hash']):
                    raise ValueError('resume output checkpoint belongs to a different run/provenance')
                if existing['epoch'] > checkpoint['epoch']:
                    raise ValueError('cannot rewind an existing run output; use a new output directory')
    code_identity = _code_identity()
    _seed(cfg['seed'])
    model = _model(cfg, initialize=resume is None).to(cfg['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    run_id, start_epoch, best = uuid.uuid4().hex, 0, -1.
    history = []
    if resume is not None:
        def comparable(config):
            return {k: v for k, v in config.items() if k not in ('epochs', 'tracking')}
        if (checkpoint['split_hash'] != hash_file(split_path) or checkpoint['dataset_hash'] != split['dataset_hash']
                or comparable(checkpoint['config']) != comparable(cfg)):
            raise ValueError('resume provenance/config mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch, best, run_id = checkpoint['epoch'], checkpoint['best_auroc'], checkpoint['run_id']
        history = checkpoint['history']
        _restore_rng(checkpoint['rng'])
        if cfg['epochs'] <= start_epoch:
            raise ValueError('resume target must exceed completed epochs')
    tracker = Tracker(output, cfg['tracking'], run_id, cfg)
    write_json(output / 'resolved-config.json', cfg)
    training = _loader(data_root, split['train_ids'], cfg, split['labels'], shuffle=True)
    validation = _loader(data_root, split['val_ids'], cfg, split['labels'])
    status = 'running'
    if cfg['device'] == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    training_started = time.monotonic()
    try:
        for epoch in range(start_epoch, cfg['epochs']):
            if cfg['device'] == 'cuda':
                torch.cuda.synchronize()
            started = time.monotonic()
            model.train()
            loss_sum = 0.
            for images, labels, _ in training:
                if cfg.get('max_runtime_seconds') is not None and time.monotonic() - training_started >= cfg['max_runtime_seconds']:
                    raise TimeoutError('training time budget exhausted at batch boundary')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=cfg['device'], enabled=cfg['amp']):
                    logits = model(images.to(cfg['device'])).flatten()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.to(cfg['device']).float())
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('nonfinite gradient')
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(images)
            if cfg['device'] == 'cuda':
                torch.cuda.synchronize()
            training_finished = time.monotonic()
            rows = _scores(model, validation, cfg['device'])
            if cfg['device'] == 'cuda':
                torch.cuda.synchronize()
            validation_finished = time.monotonic()
            score = _auroc(rows)
            metrics = {'epoch': epoch + 1, 'loss': loss_sum / len(training.dataset),
                       'val_auroc': score, 'seconds': time.monotonic() - started,
                       'train_seconds': training_finished-started,
                       'validation_seconds': validation_finished-training_finished}
            history.append(metrics)
            improved = score > best  # earliest epoch wins ties
            best = max(best, score)
            state = {'schema_version': 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict(), 'scheduler': None, 'epoch': epoch + 1,
                     'best_auroc': best, 'run_id': run_id, 'history': history, 'rng': _rng(),
                     'config': cfg, 'config_hash': hash_json(cfg), 'model_spec': _spec(cfg),
                     'environment': _environment(cfg['device']), 'code_identity': code_identity, 'train_ids': split['train_ids'], 'val_ids': split['val_ids'],
                     'labels': split['labels'],
                     'split_hash': hash_file(split_path), 'dataset_hash': split['dataset_hash']}
            _save_checkpoint(output / 'last.pt', state)
            if improved:
                _save_checkpoint(output / 'best.pt', state)
            tracker.log(metrics, step=epoch + 1)
        status = 'completed'
        result = {'schema_version': 1, 'status': status, 'run_id': run_id, 'best_auroc': best,
                  'completed_epochs': cfg['epochs'], 'history': history,
                  'best_checkpoint': str(output / 'best.pt'), 'last_checkpoint': str(output / 'last.pt'),
                  'split_hash': hash_file(split_path), 'dataset_hash': split['dataset_hash'],
                  'config_hash': hash_json(cfg), 'protocol_hash': hash_json(_spec(cfg)),
                  'metrics': {'auroc': best, 'train_seconds': sum(h['train_seconds'] for h in history),
                              'validation_seconds': sum(h['validation_seconds'] for h in history)},
                  'resources': {'execution_device': cfg['device'],
                                'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if cfg['device'] == 'cuda' else None},
                  'code_identity': code_identity}
        # A resumed run in a new directory may not improve on its previous best.
        if not (output / 'best.pt').exists() and resume is not None:
            previous_best = Path(resume).parent / 'best.pt'
            if previous_best.exists():
                prior = torch.load(previous_best, map_location='cpu', weights_only=False)
                if prior['run_id'] != run_id or prior['best_auroc'] != best:
                    raise ValueError('resume sibling best checkpoint does not match resumed run')
                import shutil
                shutil.copyfile(previous_best, output / 'best.pt')
            else:
                result['best_checkpoint'] = None
        for name in ('best.pt', 'last.pt'):
            if (output / name).exists():
                tracker.artifact(output / name, f'{run_id}-{name[:-3]}')
        write_json(output / 'model-manifest.json', {
            'schema_version': 1, 'run_id': run_id, 'config': cfg, 'config_hash': hash_json(cfg),
            'model_spec': _spec(cfg), 'protocol_hash': hash_json(_spec(cfg)),
            'dataset_hash': split['dataset_hash'], 'split_hash': hash_file(split_path),
            'train_ids': split['train_ids'], 'val_ids': split['val_ids'],
            'pretrained': cfg.get('pretrained'), 'environment': _environment(cfg['device']), 'code_identity': code_identity,
            'checkpoints': {name: {'path': str((output / name).resolve()), 'sha256': hash_file(output / name)}
                            for name in ('best.pt', 'last.pt') if (output / name).exists()}})
        write_json(output / 'run.json', result)
        return result
    except BaseException as error:
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        write_json(output / 'run.json', {'status': status, 'run_id': run_id,
                                       'error': f'{type(error).__name__}: {error}', 'history': history})
        raise
    finally:
        tracker.finish(status)


def _load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = {**checkpoint['config'], 'device': 'cpu', 'amp': False}
    if checkpoint['model_spec'] != _spec(cfg):
        raise ValueError('current inference implementation/model spec differs from checkpoint')
    model = _model(cfg)
    model.load_state_dict(checkpoint['model'])
    return model, cfg, checkpoint


def _provenance(checkpoint_path, checkpoint, kind):
    return {'schema_version': 1, 'kind': kind, 'checkpoint_path': str(Path(checkpoint_path).resolve()),
            'checkpoint_sha256': hash_file(checkpoint_path),
            'train_ids': checkpoint['train_ids'], 'val_ids': checkpoint['val_ids'],
            'split_hash': checkpoint['split_hash'], 'dataset_hash': checkpoint['dataset_hash'],
            'protocol_hash': hash_json(checkpoint['model_spec']), 'model_spec': checkpoint['model_spec'],
            'run_id': checkpoint['run_id'], 'code_identity': checkpoint['code_identity']}


def evaluate(data_root, split_path, checkpoint_path, output_path):
    split = _split(split_path)
    _output_guard(data_root, output_path)
    _validate_dataset(data_root, split['dataset_hash'], split)
    model, cfg, checkpoint = _load_model(checkpoint_path)
    if (hash_file(split_path) != checkpoint['split_hash'] or split['dataset_hash'] != checkpoint['dataset_hash']
            or split['train_ids'] != checkpoint['train_ids'] or split['val_ids'] != checkpoint['val_ids']):
        raise ValueError('evaluation split/checkpoint mismatch')
    rows = _scores(model, _loader(data_root, split['val_ids'], cfg, split['labels']), 'cpu')
    result = {**_provenance(checkpoint_path, checkpoint, 'validation'), 'rows': rows, 'auroc': _auroc(rows)}
    write_json(output_path, result)
    return result


def predict(data_root, template_path, checkpoint_path, output_path):
    _output_guard(data_root, output_path)
    with Path(template_path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['image_path', 'anomaly_score']:
            raise ValueError('submission template columns must be image_path,anomaly_score')
        ids = [row['image_path'] for row in reader]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('empty/duplicate template IDs')
    model, cfg, checkpoint = _load_model(checkpoint_path)
    _validate_dataset(data_root, checkpoint['dataset_hash'])
    if set(ids) & set(checkpoint['train_ids'] + checkpoint['val_ids']):
        raise ValueError('test IDs overlap train/validation')
    rows = _scores(model, _loader(data_root, ids, cfg, role='test'), 'cpu')
    rows = [{k: r[k] for k in ('image_path', 'anomaly_score')} for r in rows]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_path', 'anomaly_score'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    result = {**_provenance(checkpoint_path, checkpoint, 'test'), 'rows': rows,
              'csv_sha256': hash_file(output), 'template_sha256': hash_file(template_path)}
    write_json(str(output) + '.json', result)
    return result


def _prediction(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        result = load_json(str(path) + '.json')
        if hash_file(path) != result['csv_sha256']:
            raise ValueError('prediction CSV hash mismatch')
        with path.open(encoding='utf-8', newline='') as stream:
            rows = [{**r, 'anomaly_score': float(r['anomaly_score'])} for r in csv.DictReader(stream)]
        if rows != result['rows']:
            raise ValueError('prediction CSV/sidecar mismatch')
        return result
    return load_json(path)


def _checked_rows(result):
    rows = result['rows']
    ids = [r['image_path'] for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty/duplicate prediction IDs')
    if any(not math.isfinite(r['anomaly_score']) or not 0 <= r['anomaly_score'] <= 1 for r in rows):
        raise ValueError('invalid probability')
    return {r['image_path']: r for r in rows}


def _recipe_identity(recipe):
    """Portable model/aggregation identity, independent of relocated prediction evidence."""
    return {'method': recipe['method'], 'weights': recipe['weights'],
            'members': [{'checkpoint_sha256': m['checkpoint_sha256'], 'protocol_hash': m['protocol_hash']}
                        for m in recipe['members']],
            'split_hash': recipe['split_hash'], 'dataset_hash': recipe['dataset_hash'],
            'val_ids': sorted(recipe['val_ids']), 'aggregation_order': recipe['aggregation_order']}


def ensemble(member_paths, weights, output_path, recipe_path=None, frozen_recipe_path=None):
    if len(member_paths) < 2 or len(weights) != len(member_paths):
        raise ValueError('at least two members and matching weights required')
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1., abs_tol=1e-9):
        raise ValueError('weights must be nonnegative and sum to one')
    recipe_output = Path(recipe_path or str(output_path) + '.recipe.json')
    if recipe_output.exists() or Path(output_path).exists():
        raise ValueError('ensemble output/recipe already exists; preserve earlier selection evidence')
    members = [_prediction(p) for p in member_paths]
    maps = [_checked_rows(m) for m in members]
    first = members[0]
    ids = list(maps[0])
    for member, rows in zip(members, maps):
        for key in ('kind', 'split_hash', 'dataset_hash'):
            if member.get(key) != first.get(key) or not member.get(key):
                raise ValueError(f'ensemble provenance mismatch: {key}')
        for key in ('checkpoint_sha256', 'checkpoint_path', 'model_spec', 'protocol_hash', 'train_ids', 'val_ids'):
            if not member.get(key):
                raise ValueError(f'missing member provenance: {key}')
        if hash_file(member['checkpoint_path']) != member['checkpoint_sha256']:
            raise ValueError('member checkpoint hash mismatch')
        checkpoint = torch.load(member['checkpoint_path'], map_location='cpu', weights_only=False)
        for key in ('train_ids', 'val_ids', 'split_hash', 'dataset_hash', 'model_spec', 'code_identity'):
            if member[key] != checkpoint[key]:
                raise ValueError(f'checkpoint provenance mismatch: {key}')
        if hash_json(member['model_spec']) != member['protocol_hash']:
            raise ValueError('member protocol hash mismatch')
        if set(rows) != set(ids) or set(member['val_ids']) != set(first['val_ids']):
            raise ValueError('ensemble heldout/row IDs differ')
        if set(rows) & set(member['train_ids']):
            raise ValueError('ensemble training leakage')
        if first['kind'] == 'validation' and set(rows) != set(member['val_ids']):
            raise ValueError('validation rows must be exact heldout IDs')
        if first['kind'] == 'test' and set(rows) & set(member['val_ids']):
            raise ValueError('test/validation overlap')
        if first['kind'] not in ('test', 'validation'):
            raise ValueError('unsupported ensemble kind')
        if first['kind'] == 'validation' and any(rows[i].get('label') not in (0, 1) or rows[i]['label'] != maps[0][i]['label'] or rows[i]['label'] != checkpoint['labels'][i] for i in ids):
            raise ValueError('ensemble label disagreement')
    rows = [{**maps[0][i], 'anomaly_score': sum(w*m[i]['anomaly_score'] for w, m in zip(weights, maps))} for i in ids]
    recipe = {'schema_version': 1, 'method': 'probability-weighted-mean', 'weights': list(weights),
              'members': [{'prediction_path': str(Path(p).resolve()), 'prediction_sha256': hash_file(p),
                           'checkpoint_sha256': m['checkpoint_sha256'], 'checkpoint_path': m['checkpoint_path'],
                           'protocol_hash': m['protocol_hash']} for p, m in zip(member_paths, members)],
              'split_hash': first['split_hash'], 'dataset_hash': first['dataset_hash'],
              'val_ids': first['val_ids'], 'selection_attempts': 1, 'selected_on': first['kind'],
              'aggregation_order': 'sigmoid-per-model,then-weighted-probability-mean'}
    identity = _recipe_identity(recipe)
    if first['kind'] == 'test' and frozen_recipe_path is None:
        raise ValueError('test ensemble requires a frozen validation recipe')
    if frozen_recipe_path is not None:
        frozen = load_json(frozen_recipe_path)
        if frozen.get('selected_on') != 'validation':
            raise ValueError('frozen recipe must originate from validation selection')
        if frozen.get('recipe_hash') != hash_json(_recipe_identity(frozen)):
            raise ValueError('frozen recipe identity hash mismatch')
        if identity != _recipe_identity(frozen):
            raise ValueError('members/weights/protocol differ from frozen validation recipe')
        recipe['frozen_recipe_path'] = str(Path(frozen_recipe_path).resolve())
        recipe['frozen_recipe_sha256'] = hash_file(frozen_recipe_path)
        recipe['selected_on'] = 'validation'
        recipe['selection_attempts'] = frozen['selection_attempts']
    recipe['recipe_hash'] = hash_json(identity)
    result = {'schema_version': 1, 'kind': first['kind'], 'rows': rows, 'recipe': recipe,
              'recipe_hash': recipe['recipe_hash'], 'split_hash': first['split_hash'],
              'dataset_hash': first['dataset_hash'], 'val_ids': first['val_ids']}
    if first['kind'] == 'validation':
        result['auroc'] = _auroc(rows)
    write_json(output_path, result)
    write_json(recipe_output, recipe)
    return result


def reproduce(reference_path, current_path, probability_atol, auroc_atol):
    if any(not math.isfinite(t) or t < 0 for t in (probability_atol, auroc_atol)):
        raise ValueError('tolerances must be finite and nonnegative')
    reference, current = _prediction(reference_path), _prediction(current_path)
    left, right = _checked_rows(reference), _checked_rows(current)
    if set(left) != set(right):
        raise ValueError('reproduction IDs differ')
    for key in ('kind', 'dataset_hash', 'split_hash', 'protocol_hash', 'checkpoint_sha256', 'recipe_hash'):
        if reference.get(key) != current.get(key):
            raise ValueError(f'reproduction provenance mismatch: {key}')
    for image_id in left:
        if left[image_id].get('label') != right[image_id].get('label'):
            raise ValueError('reproduction labels differ')
    maximum = max(abs(left[i]['anomaly_score']-right[i]['anomaly_score']) for i in left)
    delta = abs(_auroc(list(left.values())) - _auroc(list(right.values()))) if reference['kind'] == 'validation' else None
    return {'schema_version': 1, 'passed': maximum <= probability_atol and (delta is None or delta <= auroc_atol),
            'max_probability_difference': maximum, 'auroc_difference': delta,
            'probability_atol': probability_atol, 'auroc_atol': auroc_atol,
            'reference_sha256': hash_file(reference_path), 'current_sha256': hash_file(current_path)}


def tune(data_root, split_path, plan_path, output_dir):
    plan = load_config(plan_path)
    candidates = plan['candidates']
    maximum, budget = plan['max_candidates'], plan['time_budget_seconds']
    if type(maximum) is not int or maximum < 1 or not math.isfinite(budget) or budget <= 0 or not candidates:
        raise ValueError('finite positive candidate/time limits required')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'tuning.json'
    if summary_path.exists():
        raise ValueError('tuning output already exists; preserve earlier selection attempts')
    started, results = time.monotonic(), []
    summary = {'schema_version': 1, 'plan_hash': hash_file(plan_path), 'status': 'running',
               'selection_attempts': 0, 'results': results, 'best_candidate': None,
               'budget_policy': 'check deadline between candidates and training batches; current batch/evaluation/checkpoint may finish'}
    write_json(summary_path, summary)
    for index, candidate in enumerate(candidates[:maximum]):
        if time.monotonic() - started >= budget:
            break
        destination = output / f'candidate-{index:03d}'
        result = {'index': index, 'status': 'running'}
        results.append(result)
        try:
            config = candidate.get('config', candidate) if isinstance(candidate, dict) else candidate
            config_path = destination / 'candidate-config.json'
            destination.mkdir(parents=True, exist_ok=True)
            if isinstance(config, str):
                config = load_config(Path(plan_path).parent / config)
            config = {**config, 'max_runtime_seconds': max(.001, budget - (time.monotonic()-started))}
            write_json(config_path, config)
            run = train(data_root, split_path, config_path, destination)
            result.update(status='completed', best_auroc=run['best_auroc'], run_id=run['run_id'])
            summary['selection_attempts'] += 1
            completed = [r for r in results if r['status'] == 'completed']
            summary['best_candidate'] = max(completed, key=lambda r: r['best_auroc'])['index']
        except Exception as error:
            result.update(status='failed', error=f'{type(error).__name__}: {error}')
        summary['elapsed_seconds'] = time.monotonic() - started
        write_json(summary_path, summary)
    summary.update(status='completed', elapsed_seconds=time.monotonic()-started,
                   attempted_candidates=len(results), budget_exhausted=time.monotonic()-started >= budget)
    write_json(summary_path, summary)
    return summary
