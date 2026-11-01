"""Future CPU adapter: complete text units -> public mono s16le/24 kHz PCM.

This module is not imported by production. One sentence is computed before its
PCM blocks can be emitted; it does not claim streaming inside a VITS sentence.
The existing SpeechService must continue to own queue, IDs, SSE and priority.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from typing import Callable, Iterator

import numpy as np

from .prepare_dataset import resample_pcm, sha256

PROFILE = 'AegisVoicev2.0-PiperPTBR-pilot'


class PiperUnitAdapter:
    def __init__(self, voice, *, max_characters: int = 1200, max_audio_seconds: float = 30):
        self.voice = voice
        self.max_characters = max_characters
        self.max_audio_seconds = max_audio_seconds
        self._lock = threading.Lock()

    @classmethod
    def load(cls, model: Path, config: Path, expected_sha256: str):
        # Explicit local path only; no model discovery, network or default voice.
        if sha256(model) != expected_sha256:
            raise ValueError('candidate model hash mismatch')
        import json
        import onnxruntime as ort
        from piper import PiperVoice
        from piper.config import PiperConfig
        parameters = json.loads(config.read_text())
        if parameters.get('espeak', {}).get('voice') != 'pt-br':
            raise ValueError('a verified pt-br frontend is required')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        voice = PiperVoice(session=session, config=PiperConfig.from_dict(parameters))
        return cls(voice)

    async def __call__(self, segment, cancelled: asyncio.Event):
        """SpeechService AudioAdapter signature; HTTP/SSE remain owned by it."""
        stream = self.stream_unit(segment.normalized_text, cancelled.is_set)
        def advance():
            return next(stream, None)
        try:
            while not cancelled.is_set():
                operation = asyncio.create_task(asyncio.to_thread(advance))
                try:
                    block = await asyncio.shield(operation)
                except asyncio.CancelledError:
                    cancelled.set()
                    # Wait for the one started ONNX calculation, then discard.
                    await operation
                    raise
                if block is None or cancelled.is_set():
                    break
                yield block
        finally:
            stream.close()

    def stream_unit(self, text: str, cancelled: Callable[[], bool] = lambda: False) -> Iterator[bytes]:
        if not text.strip() or len(text) > self.max_characters:
            raise ValueError('text unit empty or too long')
        # Acquire in small intervals so cancellation remains effective while
        # another request computes. No shared audio state crosses requests.
        while not self._lock.acquire(timeout=.05):
            if cancelled():
                return
        try:
            emitted_samples = 0
            for chunk in self.voice.synthesize(text):
                if cancelled():
                    return
                if chunk.sample_channels != 1:
                    raise ValueError('candidate generated non-mono audio')
                values = np.asarray(chunk.audio_float_array, dtype=np.float64).reshape(-1)
                values = resample_pcm(values, chunk.sample_rate, 24000)
                emitted_samples += len(values)
                if emitted_samples > self.max_audio_seconds * 24000:
                    raise ValueError('candidate exceeded per-unit resource limit')
                if not np.all(np.isfinite(values)) or np.max(np.abs(values), initial=0) >= 1:
                    raise ValueError('candidate output clipped/non-finite')
                pcm = np.round(values * 32768).astype('<i2').tobytes()
                for start in range(0, len(pcm), 2400):  # 50 ms transport pieces
                    if cancelled():
                        return
                    yield pcm[start:start + 2400]
        finally:
            self._lock.release()
