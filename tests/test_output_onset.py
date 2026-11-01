"""Onset mechanics: cautious decisions, bounded streaming, and exact suffixes.

Synthetic waveforms exercise the rules; they do not constitute perceptual
approval of initial consonants in the production voice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
from output_onset import OutputOnsetTrimmer


RATE = 24000


def pcm(samples) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


def zeros(ms: float) -> np.ndarray:
    return np.zeros(round(ms * RATE / 1000), dtype="<i2")


def tone(ms: float, peak: int = 4000) -> np.ndarray:
    positions = np.arange(round(ms * RATE / 1000))
    return np.rint(peak * np.sin(2 * np.pi * 173 * positions / RATE)).astype("<i2")


def render(data: bytes, sizes: list[int], **kwargs):
    trimmer = OutputOnsetTrimmer(**kwargs)
    output = []
    offset = 0
    for size in sizes:
        output.extend(trimmer.push(data[offset:offset + size]))
        offset += size
        if offset >= len(data):
            break
    output.extend(trimmer.push(data[offset:]))
    output.extend(trimmer.finish())
    return b"".join(output), trimmer


def test_removes_detected_prefix_not_a_fixed_duration_and_retains_margin() -> None:
    for initial_ms in (180, 430, 590, 910):
        source = pcm(np.concatenate((zeros(initial_ms), tone(500), zeros(180))))
        result, trimmer = render(source, [len(source)])
        removed = (initial_ms - 80) * RATE // 1000
        assert trimmer.metrics.removed_samples == removed
        assert result == source[removed * 2:]
        assert trimmer.metrics.first_nonquiet_sample == initial_ms * RATE // 1000
        assert trimmer.metrics.decision == "trimmed"
        assert trimmer.metrics.received_samples == len(source) // 2
        assert trimmer.metrics.emitted_samples + removed == len(source) // 2


def test_emits_incrementally_before_eof_and_without_waiting_for_a_second_chunk() -> None:
    trimmer = OutputOnsetTrimmer()
    assert not trimmer.push(pcm(zeros(430)))
    first = trimmer.push(pcm(tone(10)))
    assert first and len(b"".join(first)) == (80 + 10) * RATE // 1000 * 2
    assert trimmer.metrics.decision == "trimmed"
    following = pcm(np.concatenate((tone(120), zeros(500), tone(120), zeros(90))))
    assert trimmer.push(following) == [following]
    assert not trimmer.finish()


def test_exact_suffix_survives_internal_long_silence_and_expressive_tail() -> None:
    source = pcm(np.concatenate((zeros(500), tone(200), zeros(2500), tone(300), zeros(470))))
    result, trimmer = render(source, [83, 4921, 1, 6000, 7777, 13])
    assert result == source[trimmer.metrics.removed_samples * 2:]
    assert result.endswith(pcm(zeros(470)))
    assert len(result) == len(source) - 420 * RATE // 1000 * 2


@pytest.mark.parametrize("prefix_ms", [0, 10, 70, 110])
def test_short_quiet_prefix_is_preserved(prefix_ms: int) -> None:
    source = pcm(np.concatenate((zeros(prefix_ms), tone(200))))
    result, trimmer = render(source, [len(source)])
    assert result == source
    assert trimmer.metrics.removed_samples == 0
    assert trimmer.metrics.decision == "preserved-short-prefix"


@pytest.mark.parametrize("quiet_ms", [0, 5, 200, 1995, 2000, 5300])
def test_silence_alone_is_preserved_and_probe_is_bounded(quiet_ms: int) -> None:
    source = pcm(zeros(quiet_ms))
    result, trimmer = render(source, [len(source)])
    assert result == source
    assert trimmer.metrics.removed_samples == 0
    assert trimmer.buffered_samples == 0
    assert trimmer.metrics.max_buffered_samples <= 2 * RATE
    assert trimmer.metrics.decision == ("preserved-probe-limit" if quiet_ms >= 2000 else "preserved-eof")


def test_long_quiet_prefix_falls_back_and_does_not_restart_at_later_speech() -> None:
    source = pcm(np.concatenate((zeros(2300), tone(500), zeros(1000))))
    result, trimmer = render(source, [17] * 7000)
    assert result == source
    assert trimmer.metrics.decision == "preserved-probe-limit"
    assert trimmer.metrics.first_nonquiet_sample is None


def test_low_quiet_noise_can_be_removed_but_noise_uncertainty_is_retained() -> None:
    rng = np.random.default_rng(40733133)
    quiet = rng.integers(-5, 6, len(zeros(430)))
    breath = rng.integers(-31, 32, len(zeros(170)))
    source = pcm(np.concatenate((quiet, breath, tone(300))))
    result, trimmer = render(source, [len(source)])
    assert trimmer.metrics.first_nonquiet_sample == len(quiet)
    assert result == source[(len(quiet) - len(zeros(80))) * 2:]
    assert pcm(breath) in result


@pytest.mark.parametrize("weak_prefix", [tone(160, 20), np.full(len(zeros(160)), 9), tone(160, 30)])
def test_weak_fricative_or_energy_uncertainty_opens_without_deleting_it(weak_prefix: np.ndarray) -> None:
    source = pcm(np.concatenate((zeros(430), weak_prefix, tone(250))))
    result, trimmer = render(source, [len(source)])
    # The first low-level material is uncertain (RMS or peak); a later loud
    # vowel is never used to move the onset past that material.
    assert trimmer.metrics.first_nonquiet_sample == len(zeros(430))
    assert result == source[len(zeros(350)) * 2:]
    assert pcm(weak_prefix) in result


def test_quiet_ramp_before_fricative_is_covered_by_preroll() -> None:
    ramp = np.rint(np.linspace(0, 7, len(zeros(60)))).astype("<i2")
    source = pcm(np.concatenate((zeros(430), ramp, tone(100, 40), tone(200))))
    result, trimmer = render(source, [len(source)])
    assert result == source[trimmer.metrics.removed_samples * 2:]
    assert pcm(ramp) in result


def test_isolated_transient_is_preserved_and_stops_all_later_trimming() -> None:
    transient = zeros(10)
    transient[107] = 28
    source = pcm(np.concatenate((zeros(430), transient, zeros(590), tone(200))))
    result, trimmer = render(source, [len(source)])
    assert trimmer.metrics.first_nonquiet_sample == len(zeros(430))
    assert result == source[len(zeros(350)) * 2:]
    assert pcm(transient) in result


@pytest.mark.parametrize("seed", [1, 22, 40733133])
def test_arbitrary_odd_transport_chunks_produce_identical_pcm(seed: int) -> None:
    rng = np.random.default_rng(seed)
    source = pcm(np.concatenate((zeros(437), tone(500), zeros(1400), tone(600), zeros(123))))
    expected, whole = render(source, [len(source)])
    result, divided = render(source, rng.integers(1, 2001, 1000).tolist())
    assert result == expected
    assert divided.metrics == whole.metrics
    assert len(result) % 2 == 0


def test_incomplete_sample_is_never_silently_dropped() -> None:
    trimmer = OutputOnsetTrimmer()
    assert trimmer.push(b"\x01") == []
    with pytest.raises(ValueError, match="incomplete PCM16"):
        trimmer.finish()
    assert trimmer.push(b"\x00") == []
    assert trimmer.finish() == [b"\x01\x00"]
    assert trimmer.finish() == []
    with pytest.raises(ValueError, match="after finishing"):
        trimmer.push(b"\x00\x00")


def test_successive_units_keep_prior_tail_and_next_preroll_without_shared_state() -> None:
    first = pcm(np.concatenate((zeros(500), tone(250), zeros(120))))
    second = pcm(np.concatenate((zeros(600), tone(250), zeros(160))))
    first_out, one = render(first, [len(first)])
    second_out, two = render(second, [len(second)])
    assert first_out.endswith(pcm(zeros(120)))
    assert second_out.startswith(pcm(zeros(80)))
    assert one.metrics.removed_samples == len(zeros(420))
    assert two.metrics.removed_samples == len(zeros(520))
    # No cross-request audio, amplitude normalization, or deletion of the
    # preceding final pause is needed to remove only redundant openings.
    assert first_out + second_out == first[len(zeros(420)) * 2:] + second[len(zeros(520)) * 2:]


@pytest.mark.parametrize("settings", [
    {"sample_rate": 0}, {"sample_rate": True}, {"frame_ms": 0},
    {"pre_roll_ms": -1}, {"quiet_peak_pcm16": float("nan")},
    {"quiet_rms_pcm16": 25}, {"max_probe_ms": 5},
    {"pre_roll_ms": 3000}, {"quiet_peak_pcm16": 40000},
])
def test_invalid_settings_fail_explicitly(settings: dict) -> None:
    with pytest.raises(ValueError):
        OutputOnsetTrimmer(**settings)
