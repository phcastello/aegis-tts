"""ASR orchestration with a fake local model: no downloads, GPU or speech."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import dataset_content_audit as audit_tool
from training.prepare_dataset import sha256, write_pcm


@pytest.fixture
def local_audit(tmp_path, monkeypatch):
    audio = tmp_path / 'source.wav'
    write_pcm(audio, .1 * np.sin(np.arange(22050) * .05), 22050)
    dataset = tmp_path / 'dataset.jsonl'
    row = {'id': 'one', 'text': 'Ative o protocolo.', 'audio_filepath': str(audio),
           'sha256': sha256(audio), 'quality_status': 'accepted'}
    dataset.write_text(json.dumps(row) + '\n')
    model_dir = tmp_path / 'local-model'
    model_dir.mkdir()
    (model_dir / 'model.bin').write_bytes(b'fake model test only')
    calls = []
    class FakeModel:
        def __init__(self, path, **kwargs):
            calls.append(('load', path, kwargs))
        def transcribe(self, path, **kwargs):
            assert isinstance(path, np.ndarray)
            assert path.dtype == np.float32
            assert len(path) == 16000
            assert np.all(np.isfinite(path))
            calls.append(('transcribe', path, kwargs))
            segment = SimpleNamespace(text='Ative o protocolo.', start=0., end=1., avg_logprob=-.1, no_speech_prob=.01)
            return iter([segment]), SimpleNamespace(duration=1.)
    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=FakeModel))
    monkeypatch.setattr(audit_tool, 'version', lambda _: 'test-local')
    output = tmp_path / 'audit'
    arguments = ['audit', '--dataset', str(dataset), '--model-dir', str(model_dir), '--output-dir', str(output)]
    monkeypatch.setattr(sys, 'argv', arguments)
    return dataset, audio, model_dir, output, arguments, calls


def test_asr_is_local_cpu_no_vad_no_transcript_prompt_and_private(local_audit):
    _, _, _, output, _, calls = local_audit
    audit_tool.main()
    assert calls[0][2]['device'] == 'cpu'
    assert calls[0][2]['local_files_only'] is True
    settings = calls[1][2]
    assert settings['language'] == 'pt' and not settings['vad_filter']
    assert 'initial_prompt' not in settings
    provenance = json.loads((output / 'content-audit-provenance.json').read_text())
    assert provenance['complete'] and provenance['examples_checked'] == 1
    assert (output / 'content-audit.jsonl').stat().st_mode & 0o777 == 0o600
    records = [json.loads(line) for line in (output / 'content-audit.jsonl').read_text().splitlines()]
    assert records[0]['content_review'] == 'pending'
    assert records[0]['segments'][0]['avg_logprob'] == -.1
    assert records[0]['asr_input_dtype'] == 'float32'
    assert records[0]['asr_input_samples'] == 16000
    assert records[0]['asr_input_sha256'] == audit_tool.digest(calls[1][1].tobytes())
    assert provenance['input_sample_rate'] == 16000
    assert provenance['input_format'] == 'mono numpy.float32 PCM-scaled waveform'
    assert provenance['progress']['audio_seconds_checked'] == 1.


def test_resume_uses_only_bound_records_and_refuses_overwrite(local_audit):
    _, _, _, _, arguments, calls = local_audit
    audit_tool.main()
    with pytest.raises(ValueError, match='audit output exists'):
        audit_tool.main()
    arguments.append('--resume')
    audit_tool.main()
    assert sum(call[0] == 'transcribe' for call in calls) == 1


@pytest.mark.parametrize('change', ['dataset', 'model', 'audio'])
def test_resume_refuses_changed_inputs(local_audit, change):
    dataset, audio, model_dir, _, arguments, _ = local_audit
    audit_tool.main()
    arguments.append('--resume')
    if change == 'dataset':
        dataset.write_text(dataset.read_text() + '\n')
    elif change == 'model':
        (model_dir / 'model.bin').write_bytes(b'changed')
    else:
        audio.write_bytes(b'changed')
    with pytest.raises(ValueError, match='mismatch|audio changed'):
        audit_tool.main()


@pytest.mark.parametrize('authorization_args', [[], ['--owner-authorization', 'test'], ['--exclusive-gpu-window', 'test']])
def test_cuda_denied_without_both_authorization_references(local_audit, authorization_args):
    _, _, _, output, arguments, calls = local_audit
    arguments.extend(['--device', 'cuda', *authorization_args])
    with pytest.raises(SystemExit):
        audit_tool.main()
    assert not calls and not output.exists()


def test_cuda_records_explicit_window_and_guard_without_touching_real_gpu(local_audit, monkeypatch):
    from training import train_piper
    _, _, _, output, arguments, calls = local_audit
    guard_calls = []
    monkeypatch.setattr(train_piper, 'gpu_available', lambda minimum_free_mib: guard_calls.append(minimum_free_mib))
    arguments.extend(['--device', 'cuda', '--owner-authorization', 'test-owner', '--exclusive-gpu-window', 'test-window'])
    audit_tool.main()
    assert guard_calls == [9000]
    assert calls[0][2]['device'] == 'cuda' and calls[0][2]['compute_type'] == 'float16'
    provenance = json.loads((output / 'content-audit-provenance.json').read_text())
    assert provenance['device'] == 'cuda' and provenance['compute_type'] == 'float16'
    assert provenance['owner_authorization'] == 'test-owner'
    assert provenance['exclusive_gpu_window'] == 'test-window'
    assert provenance['cuda_capacity_check_passed'] is True


def test_cuda_denied_when_production_occupies_capacity(local_audit, monkeypatch):
    from training import train_piper
    _, _, _, output, arguments, calls = local_audit
    def unavailable(**kwargs):
        raise RuntimeError('exclusive GPU capacity unavailable')
    monkeypatch.setattr(train_piper, 'gpu_available', unavailable)
    arguments.extend(['--device', 'cuda', '--owner-authorization', 'test-owner', '--exclusive-gpu-window', 'test-window'])
    with pytest.raises(RuntimeError, match='capacity unavailable'):
        audit_tool.main()
    assert not calls and not output.exists()


def test_resume_never_mixes_cpu_and_cuda_records(local_audit, monkeypatch):
    from training import train_piper
    _, _, _, _, arguments, _ = local_audit
    audit_tool.main()
    monkeypatch.setattr(train_piper, 'gpu_available', lambda **kwargs: None)
    arguments.extend(['--resume', '--device', 'cuda', '--owner-authorization', 'test-owner', '--exclusive-gpu-window', 'test-window'])
    with pytest.raises(ValueError, match='settings mismatch'):
        audit_tool.main()
