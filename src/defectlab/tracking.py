"""Local-first run records; W&B is optional and imported only on request."""
import json
from pathlib import Path
from datetime import datetime, timezone


class Tracker:
    def __init__(self, output_dir, config, run_id, resolved_config):
        self.path = Path(output_dir) / 'events.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run = None
        mode = config.get('mode', 'disabled')
        self.mode = mode
        self.artifact_wait_seconds = config.get('artifact_wait_seconds', 60)
        if mode not in {'disabled', 'offline', 'online'}:
            raise ValueError('tracking.mode must be disabled, offline or online')
        if not 0 < self.artifact_wait_seconds <= 60:
            raise ValueError('artifact_wait_seconds must be in (0, 60]')
        self.log({'event': 'tracking_started', 'run_id': run_id, 'mode': mode})
        if mode != 'disabled':
            import wandb
            self.run = wandb.init(project=config.get('project', 'defectlab'),
                                  entity=config.get('entity'), id=run_id,
                                  mode=mode, dir=str(Path(output_dir).resolve()),
                                  config=resolved_config, resume='allow' if mode == 'online' else None,
                                  settings=wandb.Settings(disable_git=True, save_code=False))

    def log(self, values, step=None):
        record = {'time': datetime.now(timezone.utc).isoformat(), **values}
        if step is not None:
            record['step'] = step
        with self.path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        if self.run is not None:
            self.run.log(values, step=step)

    def artifact(self, path, name, kind='model'):
        """Record local bytes; only an acknowledged online upload has a cloud version."""
        from .common import hash_file
        receipt = {'event': 'artifact', 'path': str(path), 'sha256': hash_file(path),
                   'name': name, 'kind': kind, 'mode': self.mode,
                   'cloud_version': None, 'cloud_qualified_name': None,
                   'upload_acknowledged': False}
        # Persist intent and byte identity even when SDK upload fails.
        self.log({**receipt, 'event': 'artifact_started'})
        if self.run is not None:
            import wandb
            artifact = wandb.Artifact(name=name, type=kind, metadata={'sha256': receipt['sha256']})
            artifact.add_file(str(path))
            logged = self.run.log_artifact(artifact)
            if self.mode == 'online':
                acknowledged = logged.wait(timeout=self.artifact_wait_seconds)
                receipt.update(cloud_version=acknowledged.version,
                               cloud_qualified_name=acknowledged.qualified_name,
                               upload_acknowledged=True)
        self.log(receipt)
        return receipt

    def finish(self, status='completed'):
        self.log({'event': 'finished', 'status': status})
        if self.run is not None:
            self.run.finish(exit_code=0 if status == 'completed' else 1)
