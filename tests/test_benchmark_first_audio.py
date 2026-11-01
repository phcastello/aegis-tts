"""Measurement integrity and private artifact handling, without model traffic."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

SPEC = importlib.util.spec_from_file_location('benchmark_first_audio', Path(__file__).resolve().parents[1] / 'scripts/benchmark-first-audio.py')
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_private_artifacts_refuse_public_directory_git_and_symlinks(tmp_path):
    public = tmp_path / 'public'
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError, match='0700'):
        benchmark.private_directory(public)
    repository = tmp_path / 'checkout'
    repository.mkdir()
    (repository / '.git').write_text('gitdir: elsewhere')
    with pytest.raises(ValueError, match='Git checkout'):
        benchmark.private_directory(repository / 'audio')
    private = benchmark.private_directory(tmp_path / 'private')
    benchmark.write_private(private / 'recording.pcm', b'private samples')
    assert (private / 'recording.pcm').stat().st_mode & 0o777 == 0o600
    sentinel = tmp_path / 'sentinel'
    sentinel.write_bytes(b'keep')
    (private / 'symlink.pcm').symlink_to(sentinel)
    with pytest.raises(OSError):
        benchmark.write_private(private / 'symlink.pcm', b'replace')
    assert sentinel.read_bytes() == b'keep'


def test_receive_handles_pcm_samples_split_between_chunks(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(benchmark.time, 'perf_counter', lambda: clock[0])
    class Body(httpx.SyncByteStream):
        def __iter__(self):
            for chunk in [b'\x00', b'\x00\x65', b'\x00']:
                clock[0] += .01
                yield chunk
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Body())), base_url='http://example.test') as client:
        raw, row = benchmark.receive(client, '/v1/audio/speech', {}, 'synthetic-id')
    assert raw == b'\x00\x00\x65\x00'
    assert row['first_pcm_ms'] == pytest.approx(10)
    assert row['first_nonsilent_ms'] == pytest.approx(30)
    assert row['leading_nonsilent_audio_ms'] == pytest.approx(1 / 24)
    # The voiced sample is incomplete until the third arrival at 30ms.
    assert row['estimated_first_audible_playback_ms'] == pytest.approx(30)
    assert row['first_voice_available_ms'] == pytest.approx(30)
    assert [piece['bytes'] for piece in row['chunks']] == [1, 2, 1]


def test_paced_replay_preserves_eof_wait_for_short_finalization(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(benchmark.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(benchmark.time, 'sleep', lambda delay: clock.__setitem__(0, clock[0] + delay))
    class ShortProcessor:
        def __init__(self, **kwargs):
            self._started = clock[0]
            self.metrics = SimpleNamespace(first_render_started_seconds=None, first_render_finished_seconds=None,
                world_seconds=0, first_render_received_samples=0, hard_seams=0, natural_segments=0)
        def push(self, samples):
            return []
        def finish(self):
            return [b'\x65\x00']
    monkeypatch.setattr(benchmark, 'AcousticStreamProcessor', ShortProcessor)
    pcm, row = benchmark.render(b'\x65\x00', [{'ms': 50, 'bytes': 2}], 'stable', paced=True, eof_ms=200)
    assert pcm == b'\x65\x00'
    assert row['first_pcm_ms'] == pytest.approx(200)
    with pytest.raises(ValueError, match='does not consume'):
        benchmark.render(b'\x65\x00', [], 'stable', paced=False)
    with pytest.raises(ValueError, match='exceeds'):
        benchmark.render(b'\x65\x00', [{'ms': 0, 'bytes': 3}], 'stable', paced=False)


def test_audible_estimate_does_not_claim_silent_audio_is_audible():
    row = benchmark.audible_estimate(b'\x00\x00' * 240, 50)
    assert row['leading_nonsilent_audio_ms'] is None
    assert row['estimated_first_audible_playback_ms'] is None
    # PCM16's minimum value must be widened before taking its absolute value.
    row = benchmark.audible_estimate(b'\x00\x00' * 240 + b'\x00\x80', 50)
    assert row['leading_nonsilent_audio_ms'] == 10
    assert row['estimated_first_audible_playback_ms'] == 60


def test_audible_estimate_respects_late_voiced_batch_and_underrun():
    # 80ms of silent PCM at 300ms, followed by 420ms of silence then speech
    # in the 1400ms batch. A naive 300+500=800ms is physically impossible.
    pcm = b'\0\0' * 12000 + b'\x65\x00' * 2400
    chunks = [{'ms': 300, 'bytes': 3840}, {'ms': 1400, 'bytes': len(pcm) - 3840}]
    row = benchmark.audible_estimate(pcm, 300, chunks)
    assert row['first_voice_available_ms'] == 1400
    assert row['estimated_first_audible_playback_ms'] == 1820
    assert row['immediate_playback_underruns'] == 1
    assert row['immediate_playback_underrun_ms'] == 1020
