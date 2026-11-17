import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from defectlab.common import hash_json, load_config, safe_path, write_json


def test_canonical_hash_and_atomic_json(tmp_path):
    assert hash_json({'a': 1, 'b': 2}) == hash_json({'b': 2, 'a': 1})
    target = tmp_path / 'nested' / 'a.json'
    write_json(target, {'한글': 1})
    assert json.loads(target.read_bytes()) == {'한글': 1}
    assert not list(target.parent.glob('.a.json*'))
    with pytest.raises(ValueError):
        hash_json({'x': float('nan')})


def test_path_escape_and_toml(tmp_path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, '../outside')
    (tmp_path / 'config.toml').write_text('seed = 42\n')
    assert load_config(tmp_path / 'config.toml') == {'seed': 42}


def test_management_cli_no_torch_import(tmp_path):
    code = "from defectlab.cli import main; import sys; main(['status',sys.argv[1]]); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path / 'ledger.json')], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert result.returncode == 0, result.stderr


def test_verify_failure_sets_process_exit_code(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'defectlab', 'verify', str(tmp_path / 'missing')],
                            capture_output=True, text=True,
                            env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')})
    assert result.returncode == 1
    assert json.loads(result.stdout)['ok'] is False
