"""Small CPU probes for the GPU executor's isolation and artifact failure modes."""
import io
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest
from training import v2_voxcpm_gateway_probe as probe


def test_default_plan_does_not_probe_gpu_start_service_or_resolve_venv_interpreter(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(probe,'execute',lambda _:pytest.fail('GPU execution from default plan'))
    monkeypatch.setattr(probe,'cpu_check',lambda _:pytest.fail('Unexpected Torch import from plan'))
    monkeypatch.setattr('sys.argv',['probe','--run',str(tmp_path/'plan')])
    probe.main();value=json.loads(capsys.readouterr().out)
    assert value['gpu_executed'] is False and value['service_launched'] is False
    assert value['python']==str(probe.PYTHON)
    assert value['matrix'].startswith('seven validation')


def test_child_environment_contains_only_narrow_overlay_and_gateway(tmp_path,monkeypatch):
    monkeypatch.setenv('PYTHONPATH','/unrelated/system/torch')
    value=probe.child_environment(SimpleNamespace(overlay=tmp_path/'deps'))
    assert value['PYTHONPATH']==os.pathsep.join((str(tmp_path/'deps'),str(probe.REPO/'gateway')))
    assert value['PYTHONNOUSERSITE']=='1'


def test_complete_wav_contract_rejects_truncated_tail_and_header_only():
    output=io.BytesIO()
    with wave.open(output,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(b'\0\0'*100)
    assert probe.wav_contract(output.getvalue())['frames']==100
    for data in (output.getvalue()[:-2],output.getvalue()[:44]):
        with pytest.raises(ValueError):probe.wav_contract(data)


def test_owned_process_guard_refuses_pid_reuse_or_foreign_uid():
    sent=[]
    fake=SimpleNamespace(pid=os.getpid(),poll=lambda:None,send_signal=lambda s:sent.append(s))
    identity=probe.process_identity(fake.pid)
    with pytest.raises(RuntimeError,match='identity'):
        probe.stop_owned(fake,dict(identity,start_ticks=identity['start_ticks']+1))
    with pytest.raises(RuntimeError,match='identity'):
        probe.stop_owned(fake,dict(identity,uid=identity['uid']+1))
    assert not sent


def test_log_ring_retains_tail_without_unbounded_stdout(tmp_path):
    value=bytes(range(256))*4;path=tmp_path/'gateway.log'
    logger=probe.LimitedLog(path,limit=128,retain=32)
    logger.pump(io.BytesIO(value))
    assert path.read_bytes()==value[-32:] and logger.total_bytes==len(value) and logger.truncations==1


def test_http200_with_partial_failed_generation_is_not_a_completed_status():
    row=dict(case='v_command',repetition=0,headers={'x-aegis-request-id':'x'},outcome='ok')
    result=probe.status_receipt(row,200,json.dumps(dict(request_id='x',status='failed')).encode())
    assert result['status']=='failed' and result['request_id']=='x'
    with pytest.raises(ValueError):probe.status_receipt(row,200,b'{"request_id":"different","status":"completed"}')


def test_sigterm_runs_protected_cleanup_and_restores_previous_handlers():
    previous=signal.getsignal(signal.SIGTERM);cleaned=[]
    with pytest.raises(KeyboardInterrupt):
        with probe.cleanup_signals():
            try:signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
            finally:cleaned.append(True)
    assert cleaned==[True] and signal.getsignal(signal.SIGTERM) is previous


def test_spawn_stop_is_delivered_after_both_owner_bindings_without_blocking_child_signals():
    events=[];previous=signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        with probe.cleanup_signals():
            try:
                with probe.defer_spawn_signals():
                    events.append('process-bound')
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
                    events.append('identity-bound')
                events.append('after-stop')
            finally:events.append('owned-cleanup')
    assert events==['process-bound','identity-bound','owned-cleanup']
    assert signal.getsignal(signal.SIGTERM) is previous
