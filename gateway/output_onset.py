"""Conservative, incremental removal of a synthesized PCM16 quiet prefix.

This operates after synthesis/DSP, once per independent speech unit. It is not
microphone VAD and does not search for boundaries after the first uncertain
frame. All samples after opening, including pauses and expressive tails, pass
through unchanged. A quiet prefix alone is insufficient evidence to shorten a
stream: an entirely quiet or overlong probe is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class OnsetMetrics:
    received_samples: int = 0
    emitted_samples: int = 0
    removed_samples: int = 0
    first_nonquiet_sample: int | None = None
    max_buffered_samples: int = 0
    decision: str = "pending"


class OutputOnsetTrimmer:
    """Trim only a confidently quiet opening, retaining acoustic preroll.

    A 10 ms frame is quiet only when *both* peak and RMS are below stringent
    PCM16 thresholds. The first frame that fails either check ends the probe;
    weak fricatives, breaths, DC offsets, and isolated transients therefore
    prevent further trimming. The uncertain frame and 80 ms before it are
    retained. No gain, fade, resampling, or waveform editing is applied.

    ``push`` emits before EOF as soon as an uncertain frame arrives. Buffering
    is bounded by ``max_probe_ms`` (two seconds by default), after which the
    complete prefix is released unchanged. All-silence streams are also
    unchanged. Byte chunks may divide PCM16 samples; an incomplete final sample
    raises ``ValueError`` instead of being silently discarded.

    Recreate this object for each independent synthesized unit. Preserve the
    preceding unit's trailing audio; this leaves its pause plus the next unit's
    preroll and does not create a direct consonant-to-consonant join.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        *,
        frame_ms: float = 10.0,
        pre_roll_ms: float = 80.0,
        minimum_quiet_ms: float = 120.0,
        max_probe_ms: float = 2000.0,
        quiet_peak_pcm16: float = 24.0,
        quiet_rms_pcm16: float = 8.0,
    ) -> None:
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        values = (frame_ms, pre_roll_ms, minimum_quiet_ms, max_probe_ms,
                  quiet_peak_pcm16, quiet_rms_pcm16)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("onset settings must be finite and nonnegative")
        if frame_ms <= 0 or max_probe_ms < frame_ms or pre_roll_ms > max_probe_ms:
            raise ValueError("probe must contain a frame and the retained preroll")
        if quiet_peak_pcm16 > 32768 or quiet_rms_pcm16 > quiet_peak_pcm16:
            raise ValueError("invalid PCM16 quiet thresholds")
        self.sample_rate = sample_rate
        self.frame_samples = max(1, round(sample_rate * frame_ms / 1000))
        self.pre_roll_samples = round(sample_rate * pre_roll_ms / 1000)
        self.minimum_quiet_samples = round(sample_rate * minimum_quiet_ms / 1000)
        # End on a complete analysis frame even with an unusual sample rate.
        probe_frames = max(1, math.floor(sample_rate * max_probe_ms / 1000 / self.frame_samples))
        self.max_probe_samples = probe_frames * self.frame_samples
        if self.pre_roll_samples > self.max_probe_samples:
            raise ValueError("rounded probe must contain the retained preroll")
        self.quiet_peak_pcm16 = quiet_peak_pcm16
        self.quiet_rms_pcm16 = quiet_rms_pcm16
        self.metrics = OnsetMetrics()
        self._probe = bytearray()
        self._residual = b""
        self._opened = False
        self._finished = False

    @property
    def buffered_samples(self) -> int:
        return len(self._probe) // 2

    def push(self, chunk: bytes) -> list[bytes]:
        if self._finished:
            raise ValueError("cannot push after finishing the onset stream")
        data = self._residual + bytes(chunk)
        complete = len(data) & ~1
        self._residual = data[complete:]
        data = data[:complete]
        self.metrics.received_samples += len(data) // 2
        if not data:
            return []
        if self._opened:
            return self._emit(data)

        output: list[bytes] = []
        offset = 0
        frame_bytes = self.frame_samples * 2
        while offset < len(data) and not self._opened:
            needed = frame_bytes - len(self._probe) % frame_bytes
            end = min(offset + needed, len(data))
            self._probe.extend(data[offset:end])
            offset = end
            self.metrics.max_buffered_samples = max(self.metrics.max_buffered_samples,
                                                    self.buffered_samples)
            if len(self._probe) % frame_bytes:
                continue
            frame = np.frombuffer(self._probe[-frame_bytes:], dtype="<i2").astype(np.float64)
            peak = float(np.max(np.abs(frame)))
            rms = float(np.sqrt(np.mean(frame * frame)))
            if peak > self.quiet_peak_pcm16 or rms > self.quiet_rms_pcm16:
                first_nonquiet = self.buffered_samples - self.frame_samples
                self.metrics.first_nonquiet_sample = first_nonquiet
                remove = max(0, first_nonquiet - self.pre_roll_samples)
                if first_nonquiet < self.minimum_quiet_samples:
                    remove = 0
                self.metrics.removed_samples = remove
                decision = "trimmed" if remove else "preserved-short-prefix"
                output.extend(self._open(decision, remove))
            elif self.buffered_samples >= self.max_probe_samples:
                output.extend(self._open("preserved-probe-limit", 0))
        if offset < len(data):
            output.extend(self._emit(data[offset:]))
        return output

    def finish(self) -> list[bytes]:
        if self._finished:
            return []
        if self._residual:
            raise ValueError("onset PCM stream ended with an incomplete PCM16 sample")
        self._finished = True
        if self._opened:
            return []
        return self._open("preserved-eof", 0)

    def _open(self, decision: str, remove: int) -> list[bytes]:
        self._opened = True
        self.metrics.decision = decision
        kept = bytes(self._probe[remove * 2:])
        self._probe.clear()
        return self._emit(kept)

    def _emit(self, data: bytes) -> list[bytes]:
        if not data:
            return []
        self.metrics.emitted_samples += len(data) // 2
        return [data]
