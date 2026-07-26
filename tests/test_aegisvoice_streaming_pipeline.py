"""Contract tests for transport-independent AegisVoicev1.0 DSP."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gateway"))
from voice_pipeline import (  # noqa: E402
    AcousticStreamProcessor,
    Pcm16Reassembler,
    PcmProtocolError,
    StatefulM02,
    StreamingVoiceState,
    _transform_f0,
)


def _render_in_chunks(pcm: bytes, chunks: list[int]) -> bytes:
    reassembler = Pcm16Reassembler()
    processor = AcousticStreamProcessor(24000)
    output: list[bytes] = []
    offset = 0
    for size in chunks:
        if offset >= len(pcm):
            break
        output.extend(processor.push(reassembler.feed(pcm[offset : offset + size])))
        offset += size
    if offset < len(pcm):
        output.extend(processor.push(reassembler.feed(pcm[offset:])))
    reassembler.finish()
    output.extend(processor.finish())
    return b"".join(output)


def test_pcm_reassembler_preserves_odd_bytes_and_rejects_a_trailing_byte() -> None:
    reassembler = Pcm16Reassembler()
    first = reassembler.feed(b"\x01")
    second = reassembler.feed(b"\x02\x03\x04")
    assert first.size == 0
    assert second.tolist() == pytest.approx([513 / 32768, 1027 / 32768])
    reassembler.finish()

    reassembler.feed(b"\x01")
    with pytest.raises(PcmProtocolError):
        reassembler.finish()


def test_world_stream_is_independent_of_http_chunk_boundaries() -> None:
    sample_rate = 24000
    samples = 0.20 * np.sin(2 * np.pi * 180 * np.arange(sample_rate * 4) / sample_rate)
    pcm = np.rint(samples * 32767).astype("<i2").tobytes()
    one_chunk = _render_in_chunks(pcm, [len(pcm)])
    rng = np.random.default_rng(40733133)
    random_chunks = rng.integers(1, 4097, size=1000).tolist()
    assert _render_in_chunks(pcm, random_chunks) == one_chunk


def test_quantized_note_cannot_change_inside_one_voiced_region() -> None:
    state = StreamingVoiceState()
    first = _transform_f0(np.array([110.0, 115.0, 121.0]), state, 24000)
    second = _transform_f0(np.array([147.0, 165.0]), state, 24000)
    assert np.unique(np.concatenate((first, second))).size == 1
    _transform_f0(np.array([0.0]), state, 24000)
    after_break = _transform_f0(np.array([165.0]), state, 24000)
    assert after_break[0] != first[0]


def test_stateful_m02_matches_whole_file_processing() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(0, 0.08, 24000).astype(np.float64)
    whole = StatefulM02(24000).process(source)
    block_chain = StatefulM02(24000)
    blocked = np.concatenate([block_chain.process(source[offset : offset + 733]) for offset in range(0, len(source), 733)])
    np.testing.assert_allclose(blocked, whole, rtol=0, atol=1e-12)
