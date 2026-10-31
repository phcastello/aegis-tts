"""Critical-path timing, cancellation and explicit acoustic-variant contracts."""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gateway'))
import default_reference_gateway as gateway
from aegis_speech.domain import SegmentStatus, SpeechSegment
from voice_pipeline import AcousticStreamProcessor, Pcm16Reassembler


def test_initial_segment_is_explicit_and_only_changes_first_boundary() -> None:
    stable = AcousticStreamProcessor()
    initial = AcousticStreamProcessor(initial_segment='initial-quiet-900ms-v1')
    initial._render_world = lambda source, **kwargs: source
    assert initial.left_context == stable.left_context
    assert initial.right_lookahead == stable.right_lookahead
    # Stable waits for 1.75 s of samples; the named variant waits for 1.25 s.
    assert not stable.push(np.ones(30000) * .05)
    source = np.ones(30000) * .05
    source[18000:20000] = 0
    assert initial.push(source)
    assert initial._segment_limits() == stable._segment_limits()
    with pytest.raises(ValueError):
        AcousticStreamProcessor(initial_segment='unreviewed')


def test_early_segment_never_forces_a_new_seam() -> None:
    processor = AcousticStreamProcessor(initial_segment='initial-quiet-900ms-v1')
    processor._render_world = lambda source, **kwargs: source
    assert not processor.push(np.ones(30000) * .05)
    assert processor.push(np.ones(12000) * .05)
    assert processor.metrics.first_render_received_samples == 42000


def _render(pcm: bytes, sizes: list[int], variant: str) -> bytes:
    processor = AcousticStreamProcessor(initial_segment=variant)
    reassembler = Pcm16Reassembler()
    output, offset = [], 0
    for size in sizes:
        output.extend(processor.push(reassembler.feed(pcm[offset:offset + size])))
        offset += size
        if offset >= len(pcm): break
    output.extend(processor.push(reassembler.feed(pcm[offset:])))
    reassembler.finish()
    output.extend(processor.finish())
    return b''.join(output)


@pytest.mark.parametrize('variant', ['stable', 'initial-quiet-900ms-v1'])
@pytest.mark.parametrize('duration', [.025, .3, 4.0])
def test_variants_preserve_transport_fragmentation_and_finalization(variant: str, duration: float) -> None:
    # A synthetic voiced signal with silence and unvoiced-like noise exercises
    # natural and hard boundaries. This verifies mechanics, not voice approval.
    rng = np.random.default_rng(40733133)
    samples = .16 * np.sin(2 * np.pi * 160 * np.arange(round(24000 * duration)) / 24000)
    samples += rng.normal(0, .002, len(samples))
    if duration > 1:
        samples[18000:20000] = 0
        samples[50000:54000] = rng.normal(0, .03, 4000)
    pcm = np.rint(samples * 32767).astype('<i2').tobytes()
    whole = _render(pcm, [len(pcm)], variant)
    divided = _render(pcm, rng.integers(1, 4096, 500).tolist(), variant)
    assert divided == whole
    assert len(whole) % 2 == 0
    assert abs(len(whole) - len(pcm)) <= round(24000 * .03) * 2
    assert max(abs(np.frombuffer(whole, dtype='<i2').astype(np.int32)), default=0) < 32767


@pytest.mark.asyncio
async def test_cpu_dsp_does_not_block_event_loop_and_joins_on_cancel() -> None:
    started, finished = threading.Event(), threading.Event()
    def dsp() -> bytes:
        started.set()
        time.sleep(.06)
        finished.set()
        return b'\0\0'
    task = asyncio.create_task(gateway._dsp_call(dsp))
    while not started.is_set(): await asyncio.sleep(.001)
    assert not finished.is_set()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_adapter_emits_useful_pcm_before_eof_and_discards_after_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = np.rint(.15 * np.sin(2 * np.pi * 160 * np.arange(72000) / 24000) * 32767).astype('<i2').tobytes()
    cancelled = asyncio.Event()
    state = {'eof': False, 'closed': False}
    class Upstream:
        is_error = False
        async def aiter_bytes(self):
            # One large, odd-sized transport block must not postpone all DSP
            # until EOF; the residual sample spans the next block.
            yield samples[:-1]
            state['eof'] = True
            yield samples[-1:]
        async def aclose(self): state['closed'] = True
    upstream = Upstream()
    class Client:
        def build_request(self, *args, **kwargs): return None
        async def send(self, *args, **kwargs): return upstream
    monkeypatch.setattr(gateway, '_apply_defaults', lambda p: p)
    monkeypatch.setattr(gateway, 'INITIAL_SEGMENT', 'stable')
    monkeypatch.setattr(gateway, 'PROFILE_LATENCY', False)
    monkeypatch.setattr(AcousticStreamProcessor, '_render_world', lambda self, source, **kwargs: source)
    gateway.app.state.client = Client()
    gateway.app.state.metrics = gateway.GatewayMetrics()
    segment = SpeechSegment('0' * 26, 0, 'synthetic', 'synthetic', SegmentStatus.QUEUED, 1)
    stream = gateway._segment_adapter(segment, cancelled)
    first = await anext(stream)
    assert first and np.max(np.abs(np.frombuffer(first, dtype='<i2').astype(np.int32))) > 100
    assert not state['eof']
    cancelled.set()
    assert [chunk async for chunk in stream] == []
    assert state['closed']
    assert gateway.app.state.metrics.values['aegis_tts_first_raw_pcm_seconds'] > 0
    assert gateway.app.state.metrics.values['aegis_tts_first_processed_pcm_seconds'] > 0
