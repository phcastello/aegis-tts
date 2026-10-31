"""Acoustic, chunk-independent processing for AegisVoicev1.0 PCM streams.

The public vLLM stream is a transport stream, not an audio segmentation
stream.  This module deliberately keeps those two concerns separate: arbitrary
HTTP byte chunks are first reassembled into PCM16 samples and only then are
acoustic boundaries selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter, thread_time
from typing import Iterable

import numpy as np
from scipy.signal import butter, sosfilt


PCM_SCALE = 32768.0
W04_STEP_CENTS = 200.0
W04_FORMANT_SCALE = 1.10
P07_VOICED_APERIODICITY = 0.52
M02_PEAK = 10.0 ** (-1.0 / 20.0)


class PcmProtocolError(ValueError):
    """The backend ended an s16le stream halfway through a sample."""


class Pcm16Reassembler:
    """Reassemble arbitrary byte chunks without ever dropping an odd byte."""

    def __init__(self) -> None:
        self._residual = b""
        self.received_bytes = 0

    def feed(self, chunk: bytes) -> np.ndarray:
        self.received_bytes += len(chunk)
        data = self._residual + chunk
        complete = len(data) & ~1
        self._residual = data[complete:]
        if not complete:
            return np.empty(0, dtype=np.float64)
        # ``<i2`` is explicit: the public PCM contract is little endian.
        return np.frombuffer(data[:complete], dtype="<i2").astype(np.float64) / PCM_SCALE

    def finish(self) -> None:
        if self._residual:
            raise PcmProtocolError("backend PCM stream ended with an incomplete PCM16 sample")


@dataclass
class StreamingVoiceState:
    """Pitch state that intentionally survives WORLD analysis windows."""

    voiced_active: bool = False
    voiced_region_id: int = 0
    quantized_target_cents: float | None = None
    previous_f0_hz: float = 0.0
    unvoiced_samples: int = 0


@dataclass
class PipelineMetrics:
    audio_received_samples: int = 0
    audio_processed_samples: int = 0
    audio_emitted_samples: int = 0
    natural_segments: int = 0
    hard_seams: int = 0
    invalid_residual_bytes: int = 0
    segment_rtf: list[float] = field(default_factory=list)
    first_render_received_samples: int = 0
    first_render_started_seconds: float | None = None
    first_render_finished_seconds: float | None = None
    world_seconds: float = 0.0
    # Same-thread CPU in WORLD and render/master operations, respectively;
    # transport waiting, buffer copies and boundary selection are excluded.
    world_cpu_seconds: float = 0.0
    dsp_cpu_seconds: float = 0.0
    first_render_cpu_seconds: float | None = None

    @property
    def mean_rtf(self) -> float:
        return float(np.mean(self.segment_rtf)) if self.segment_rtf else 0.0


class StatefulM02:
    """The protected M02 chain, with filter and detector state across chunks."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._hpf = butter(2, 60, fs=sample_rate, btype="highpass", output="sos")
        self._band = butter(2, [5400, 7600], fs=sample_rate, btype="bandpass", output="sos")
        self._hpf_zi = np.zeros((self._hpf.shape[0], 2), dtype=np.float64)
        self._band_zi = np.zeros((self._band.shape[0], 2), dtype=np.float64)
        self._envelope = 0.0
        self._attack = float(np.exp(-1.0 / max(sample_rate * 0.002, 1.0)))
        self._release = float(np.exp(-1.0 / max(sample_rate * 0.045, 1.0)))

    def process(self, samples: np.ndarray) -> np.ndarray:
        if not len(samples):
            return np.empty(0, dtype=np.float64)
        high_passed, self._hpf_zi = sosfilt(self._hpf, samples, zi=self._hpf_zi)
        band, self._band_zi = sosfilt(self._band, high_passed, zi=self._band_zi)
        envelope_values: list[float] = []
        previous = self._envelope
        # Native doubles avoid NumPy scalar dispatch in this sequential
        # detector. Keep the same operation order and IEEE double precision;
        # the filter, detector state and resulting PCM remain unchanged.
        for level in memoryview(np.abs(band)):
            coefficient = self._attack if level > previous else self._release
            previous = coefficient * previous + (1.0 - coefficient) * level
            envelope_values.append(previous)
        envelope = np.asarray(envelope_values, dtype=np.float64)
        self._envelope = previous
        threshold = 10.0 ** (-30.0 / 20.0)
        excess_db = np.maximum(0.0, 20.0 * np.log10(np.maximum(envelope, 1e-12) / threshold))
        reduction_db = np.minimum(1.5, 0.75 * excess_db)
        gain = 10.0 ** (-reduction_db / 20.0)
        # This exactly mirrors the protected complementary-band de-esser.
        mastered = high_passed - band * (1.0 - gain)
        return np.clip(mastered, -M02_PEAK, M02_PEAK)

    def flush(self) -> np.ndarray:
        # IIR state has no intended audio delay; flushing therefore emits no
        # artificial tail and avoids changing response duration.
        return np.empty(0, dtype=np.float64)


def _warp_formants(spectral_envelope: np.ndarray) -> np.ndarray:
    bins = spectral_envelope.shape[1]
    positions = np.arange(bins)
    source = np.clip(positions / W04_FORMANT_SCALE, 0, bins - 1)
    return np.array([np.interp(source, positions, row) for row in spectral_envelope])


def _transform_f0(f0: np.ndarray, state: StreamingVoiceState, sample_rate: int) -> np.ndarray:
    """W04's fixed quantizer, without resetting a note at a segment edge."""
    out = np.zeros_like(f0, dtype=np.float64)
    frame_samples = max(1, round(sample_rate * 0.005))
    for index, value in enumerate(f0):
        if value > 0.0:
            if not state.voiced_active:
                state.voiced_active = True
                state.voiced_region_id += 1
                cents = 1200.0 * np.log2(value / 55.0)
                state.quantized_target_cents = np.round(cents / W04_STEP_CENTS) * W04_STEP_CENTS
            assert state.quantized_target_cents is not None
            out[index] = 55.0 * 2.0 ** (state.quantized_target_cents / 1200.0)
            state.previous_f0_hz = out[index]
            state.unvoiced_samples = 0
        else:
            state.voiced_active = False
            state.quantized_target_cents = None
            state.previous_f0_hz = 0.0
            state.unvoiced_samples += frame_samples
    return out


class AcousticStreamProcessor:
    """Segmented WORLD W04/P07 followed by one continuous M02 chain."""

    def __init__(self, sample_rate: int = 24000, *, initial_segment: str = "stable") -> None:
        if initial_segment not in {"stable", "initial-quiet-900ms-v1"}:
            raise ValueError("unsupported initial acoustic segment")
        self.sample_rate = sample_rate
        self.initial_segment = initial_segment
        self.left_context = round(sample_rate * 0.200)
        self.right_lookahead = round(sample_rate * 0.200)
        self.minimum = round(sample_rate * 0.900)
        self.target = round(sample_rate * 1.250)
        self.maximum = round(sample_rate * 2.000)
        self.search_radius = round(sample_rate * 0.300)
        self._samples = np.empty(0, dtype=np.float64)
        self._next_start = 0
        self._state = StreamingVoiceState()
        self._master = StatefulM02(sample_rate)
        self._held_tail = np.empty(0, dtype=np.float64)
        self._previous_hard = False
        self.metrics = PipelineMetrics()
        self._started = perf_counter()
        self._first_segment = True

    @property
    def buffered_seconds(self) -> float:
        return max(0, len(self._samples) - self._next_start) / self.sample_rate

    def push(self, samples: np.ndarray) -> list[bytes]:
        if len(samples):
            self._samples = np.concatenate((self._samples, samples))
            self.metrics.audio_received_samples += len(samples)
        return self._drain(final=False)

    def finish(self) -> list[bytes]:
        output = self._drain(final=True)
        tail_cpu_started = thread_time()
        if len(self._held_tail):
            output.append(self._encode(self._master.process(self._held_tail)))
            self._held_tail = np.empty(0, dtype=np.float64)
        tail = self._master.flush()
        if len(tail):
            output.append(self._encode(tail))
        self.metrics.dsp_cpu_seconds += thread_time() - tail_cpu_started
        return output

    def process_offline(self, samples: np.ndarray) -> bytes:
        """Golden path: one full WORLD analysis and M02 pass, no segmentation."""
        source = np.asarray(samples, dtype=np.float64)
        self.metrics.audio_received_samples += len(source)
        # WORLD has no samples before t=0 to stabilize DIO/StoneMask.  Give it
        # the same acoustic padding used by the first streaming segment and
        # crop it before mastering, so initial consonants are never consumed by
        # analysis startup.
        padded = np.pad(source, (self.left_context, self.right_lookahead))
        rendered = self._render_world(
            padded, useful_start=self.left_context, useful_length=len(source)
        )[self.left_context : self.left_context + len(source)]
        self.metrics.audio_processed_samples += len(source)
        return self._encode(self._master.process(rendered))

    def _drain(self, *, final: bool) -> list[bytes]:
        output: list[bytes] = []
        while True:
            available = len(self._samples) - self._next_start
            if available <= 0:
                break
            # Wait for the whole fixed boundary-search range.  This makes the
            # selected boundary independent of HTTP chunk sizes and keeps the
            # first processed audio within the 1.3--2.0 s target.
            _, target, search_radius = self._segment_limits()
            if not final and available < target + self.right_lookahead + search_radius:
                break
            useful_length, natural = self._choose_segment(available, final=final)
            if useful_length <= 0:
                break
            start, end = self._next_start, self._next_start + useful_length
            analysis_start = max(0, start - self.left_context)
            analysis_end = min(len(self._samples), end + self.right_lookahead)
            left_padding = max(0, self.left_context - start)
            right_padding = self.right_lookahead if final and analysis_end == len(self._samples) else 0
            analysis_samples = self._samples[analysis_start:analysis_end]
            if left_padding or right_padding:
                analysis_samples = np.pad(analysis_samples, (left_padding, right_padding))
            render_started = perf_counter()
            render_cpu_started = thread_time()
            if self.metrics.first_render_started_seconds is None:
                self.metrics.first_render_started_seconds = render_started - self._started
                self.metrics.first_render_received_samples = self.metrics.audio_received_samples
            rendered = self._render_world(
                analysis_samples,
                useful_start=left_padding + start - analysis_start,
                useful_length=useful_length,
            )
            useful_start = left_padding + start - analysis_start
            useful = rendered[useful_start : useful_start + useful_length]
            self.metrics.audio_processed_samples += useful_length
            # WORLD timing is measured around each complete analysis/render.
            elapsed = max(perf_counter() - render_started, 0.0)
            self.metrics.world_seconds += elapsed
            self.metrics.world_cpu_seconds += thread_time() - render_cpu_started
            self.metrics.segment_rtf.append(elapsed / max(useful_length / self.sample_rate, 1e-9))
            if natural:
                self.metrics.natural_segments += 1
            else:
                self.metrics.hard_seams += 1
            output.extend(self._join_and_master(useful, hard=not natural))
            render_cpu_seconds = thread_time() - render_cpu_started
            self.metrics.dsp_cpu_seconds += render_cpu_seconds
            if self.metrics.first_render_finished_seconds is None:
                self.metrics.first_render_finished_seconds = perf_counter() - self._started
                self.metrics.first_render_cpu_seconds = render_cpu_seconds
            self._first_segment = False
            self._next_start = end
            self._compact()
            if final and len(self._samples) - self._next_start == 0:
                break
        return output

    def _segment_limits(self) -> tuple[int, int, int]:
        # This is an explicitly named experimental variant. The 200 ms
        # analysis context and every subsequent segment keep the approved
        # settings; no voice/model/sampling/profile asset is changed.
        if self._first_segment and self.initial_segment == "initial-quiet-900ms-v1":
            return round(self.sample_rate * .750), round(self.sample_rate * .900), round(self.sample_rate * .150)
        return self.minimum, self.target, self.search_radius

    def _choose_segment(self, available: int, *, final: bool) -> tuple[int, bool]:
        if final and available <= self.maximum:
            return available, True
        upper = min(self.maximum, available - (0 if final else self.right_lookahead))
        minimum, target, search_radius = self._segment_limits()
        if self._first_segment and self.initial_segment == "initial-quiet-900ms-v1":
            # An early forced seam can cut a word and changes the amount of
            # blending. Only advance a natural quiet boundary. If none exists,
            # wait for and use exactly the stable first-window decision.
            begin = minimum
            finish = min(upper, target + search_radius)
            candidate = self._quiet_boundary(begin, finish, desired=target)
            if candidate is not None:
                return candidate, True
            if not final and available < self.target + self.right_lookahead + self.search_radius:
                return 0, True
            minimum, target, search_radius = self.minimum, self.target, self.search_radius
        if upper < minimum:
            return (available, True) if final else (0, True)
        lower = minimum
        desired = min(max(target, lower), upper)
        begin = max(lower, desired - search_radius)
        finish = min(upper, desired + search_radius)
        # A 35 ms quiet region is a usable silence/unvoiced boundary.  The
        # threshold is relative to the current useful window so quiet speech
        # is not mistaken for digital silence.
        candidate = self._quiet_boundary(begin, finish, desired=target)
        if candidate is not None:
            return candidate, True
        return self._zero_crossing_boundary(begin, finish, desired), False

    def _quiet_boundary(self, begin: int, finish: int, *, desired: int | None = None) -> int | None:
        frame = max(1, round(self.sample_rate * 0.035))
        source = self._samples[self._next_start : self._next_start + finish]
        if len(source) < frame:
            return None
        reference = np.sqrt(np.mean(source * source)) + 1e-12
        threshold = min(0.012, reference * 0.12)
        candidates: list[int] = []
        for end in range(begin, finish + 1, max(1, frame // 3)):
            window = self._samples[self._next_start + end - frame : self._next_start + end]
            if np.sqrt(np.mean(window * window)) <= threshold:
                candidates.append(end)
        if not candidates:
            return None
        desired = min(max(self.target if desired is None else desired, begin), finish)
        return min(candidates, key=lambda value: abs(value - desired))

    def _zero_crossing_boundary(self, begin: int, finish: int, desired: int) -> int:
        values = self._samples[self._next_start + begin : self._next_start + finish]
        crossings = np.flatnonzero(values[:-1] * values[1:] <= 0.0)
        if len(crossings):
            return begin + int(crossings[np.argmin(np.abs(crossings - (desired - begin)))]) + 1
        return desired

    def _render_world(
        self, samples: np.ndarray, *, useful_start: int = 0, useful_length: int | None = None
    ) -> np.ndarray:
        if not len(samples):
            return np.empty(0, dtype=np.float64)
        # Importing here keeps failures explicit at readiness time while making
        # this module inexpensive to inspect in tooling without DSP installed.
        import pyworld

        if len(samples) < round(self.sample_rate * 0.040):
            return np.zeros_like(samples)
        raw_f0, timestamps = pyworld.dio(samples.astype(np.float64), self.sample_rate, frame_period=5.0)
        f0 = pyworld.stonemask(samples.astype(np.float64), raw_f0, timestamps, self.sample_rate)
        spectral = pyworld.cheaptrick(samples.astype(np.float64), f0, timestamps, self.sample_rate)
        aperiodicity = pyworld.d4c(samples.astype(np.float64), f0, timestamps, self.sample_rate)
        voiced = f0 > 0.0
        if useful_length is None:
            useful_length = len(samples)
        hop = max(1, round(self.sample_rate * 0.005))
        first_frame = max(0, useful_start // hop)
        last_frame = min(len(f0), int(np.ceil((useful_start + useful_length) / hop)))
        # Render from a snapshot: lookahead must never alter the state of the
        # next real segment.  Only frames whose audio is actually emitted are
        # committed to the persistent voice state.
        render_state = StreamingVoiceState(**vars(self._state))
        transformed_f0 = np.zeros_like(f0, dtype=np.float64)
        transformed_f0[first_frame:] = _transform_f0(f0[first_frame:], render_state, self.sample_rate)
        _transform_f0(f0[first_frame:last_frame], self._state, self.sample_rate)
        transformed_spectral = _warp_formants(spectral)
        transformed_ap = np.asarray(aperiodicity, dtype=np.float64).copy()
        transformed_ap[voiced] *= P07_VOICED_APERIODICITY
        transformed_ap = np.clip(transformed_ap, 0.0, 1.0)
        rendered = pyworld.synthesize(transformed_f0, transformed_spectral, transformed_ap, self.sample_rate, 5.0)
        return np.pad(rendered[: len(samples)], (0, max(0, len(samples) - len(rendered))))

    def _join_and_master(self, useful: np.ndarray, *, hard: bool) -> Iterable[bytes]:
        if not len(useful):
            return []
        if not len(self._held_tail):
            if hard:
                return self._hold_tail(useful)
            return [self._encode(self._master.process(useful))]
        if not self._previous_hard:
            prefix = self._master.process(self._held_tail)
            self._held_tail = np.empty(0, dtype=np.float64)
            return [self._encode(prefix), *self._hold_tail(useful)] if hard else [self._encode(prefix), self._encode(self._master.process(useful))]
        # Hard seam: retain duration, choose the low-amplitude zero-crossing
        # boundary above, then equal-power blend only ten milliseconds.
        width = min(round(self.sample_rate * 0.010), len(self._held_tail), len(useful))
        if not width:
            joined = np.concatenate((self._held_tail, useful))
            self._held_tail = np.empty(0, dtype=np.float64)
            return [self._encode(self._master.process(joined))]
        ramp = np.linspace(0.0, np.pi / 2.0, width, endpoint=True)
        blend = self._held_tail[-width:] * np.cos(ramp) + useful[:width] * np.sin(ramp)
        prefix = self._held_tail[:-width]
        committed = np.concatenate((prefix, blend))
        self._held_tail = np.empty(0, dtype=np.float64)
        result = [self._encode(self._master.process(committed))] if len(committed) else []
        if hard:
            result.extend(self._hold_tail(useful[width:]))
        else:
            result.append(self._encode(self._master.process(useful[width:])))
        return result

    def _hold_tail(self, samples: np.ndarray) -> list[bytes]:
        width = min(round(self.sample_rate * 0.010), len(samples))
        if not width:
            return []
        committed, self._held_tail = samples[:-width], samples[-width:]
        self._previous_hard = True
        return [self._encode(self._master.process(committed))] if len(committed) else []

    def _compact(self) -> None:
        discard = max(0, self._next_start - self.left_context)
        if discard:
            self._samples = self._samples[discard:]
            self._next_start -= discard

    def _encode(self, samples: np.ndarray) -> bytes:
        if not len(samples):
            return b""
        encoded = np.clip(np.rint(samples * PCM_SCALE), -32768, 32767).astype("<i2").tobytes()
        self.metrics.audio_emitted_samples += len(samples)
        return encoded
