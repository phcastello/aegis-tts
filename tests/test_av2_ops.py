"""Operational safeguards: original deployment and finally-based rollback."""
import copy
import json
from unittest.mock import Mock

import pytest

from training import av2_ops as ops


def sample():
    return {'Id': 'original', 'Image': 'sha256:original', 'Name': '/aegis-tts', 'Config': {'Env': ['unchanged=1']}, 'HostConfig': {'Binds': ['ref:ro']}, 'Mounts': [{'Destination': '/a', 'RW': False}, {'Destination': '/b', 'RW': True}], 'State': {'Running': True}}


def test_identity_ignores_mount_listing_order_but_checks_readonly():
    first = sample()
    other = copy.deepcopy(first)
    other['Mounts'].reverse()
    assert ops.signature(first) == ops.signature(other)
    other['Mounts'][0]['RW'] = False
    assert ops.signature(first) != ops.signature(other)


def test_failure_in_child_restores_original_in_finally(tmp_path, monkeypatch):
    directory = tmp_path / 'operations'
    directory.mkdir()
    monkeypatch.setattr(ops, 'assert_original', lambda p: sample())
    monkeypatch.setattr(ops, 'gpu_pids', lambda: '')
    monkeypatch.setattr(ops.subprocess, 'run', Mock())
    child = Mock(pid=99999)
    child.wait.return_value = 23
    watcher = Mock(pid=99998)
    monkeypatch.setattr(ops.subprocess, 'Popen', Mock(side_effect=[watcher, child]))
    terminate = Mock()
    restore = Mock()
    monkeypatch.setattr(ops, 'terminate_group', terminate)
    monkeypatch.setattr(ops, 'restore', restore)
    with pytest.raises(ops.subprocess.CalledProcessError):
        ops.guarded(directory, 'http://127.0.0.1:8001', ['training-and-evaluation'])
    terminate.assert_called_once_with(child)
    restore.assert_called_once_with(directory, 'http://127.0.0.1:8001')
    journal = json.loads((directory / 'gpu-window.json').read_text())
    assert journal['training_deadline'] is None
    assert journal['child_returncode'] == 23
    assert journal['status'] == 'production_restored'
    assert json.loads((tmp_path / 'gpu-window-authorized.json').read_text())['status'] == 'gpu_phases_closed'


def test_changed_configuration_refuses_before_stop(tmp_path, monkeypatch):
    directory = tmp_path / 'operations'
    directory.mkdir()
    original = sample()
    (directory / 'snapshot.json').write_text(json.dumps(original))
    current = copy.deepcopy(original)
    current['Config']['Env'] = ['different=1']
    monkeypatch.setattr(ops, 'inspect', lambda c: current)
    docker_run = Mock()
    monkeypatch.setattr(ops.subprocess, 'run', docker_run)
    with pytest.raises(RuntimeError, match='configuration changed'):
        ops.guarded(directory, 'http://127.0.0.1:8001', ['experiment'])
    docker_run.assert_not_called()


def test_engine_accepts_actual_operations_authorization(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from training import f5_engine
    (tmp_path / 'gpu-window-authorized.json').write_text(json.dumps({
        'owner_authorized': True, 'production_stopped': True,
        'exclusive_aegisvoice_v2_gpu': True, 'status': 'active',
        'training_deadline': None}))
    monkeypatch.setattr(f5_engine.subprocess, 'check_output', lambda *a, **kw: '')
    f5_engine.require_gpu_authorization(SimpleNamespace(cuda_authorized=True),tmp_path)
