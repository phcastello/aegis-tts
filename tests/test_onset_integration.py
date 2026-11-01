"""Exercise output onset trimming through the production adapter and APIs.

The codec input is deterministic synthetic PCM and DSP is an identity worker;
these tests cover routing, isolation, streaming, EOF, and cancellation without
calling a GPU or claiming auditory validation.
"""
from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path
from types import SimpleNamespace
import wave

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
import default_reference_gateway as gateway
from aegis_speech.domain import SegmentStatus, SpeechRequest, SpeechSegment
from aegis_speech.service import SpeechService
from output_onset import OutputOnsetTrimmer
from voice_pipeline import PipelineMetrics


RATE = 24000


def quiet(ms: int) -> np.ndarray:
    return np.zeros(ms * RATE // 1000, dtype="<i2")


def tone(ms: int, frequency: int = 173) -> np.ndarray:
    samples = np.arange(ms * RATE // 1000)
    return np.rint(4000 * np.sin(2 * np.pi * frequency * samples / RATE)).astype("<i2")


def unit(initial_ms: int = 430, frequency: int = 173) -> bytes:
    # Internal silence is deliberately longer than the external prefix.
    return np.concatenate((quiet(initial_ms), tone(120, frequency), quiet(650),
                           tone(90, frequency), quiet(180))).astype("<i2").tobytes()


def segment(identifier: str = "0" * 26, index: int = 0, text: str = "unit") -> SpeechSegment:
    return SpeechSegment(identifier, index, text, text, SegmentStatus.QUEUED, 1)


class Upstream:
    is_error = False

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.eof = False
        self.closed = False

    async def aiter_bytes(self):
        # Deliberately divide a PCM sample across HTTP chunks.
        yield self.data[:7001]
        yield self.data[7001:]
        self.eof = True

    async def aclose(self) -> None:
        self.closed = True


def install(monkeypatch: pytest.MonkeyPatch, data_by_text: dict[str, bytes], *, deferred: bool = False):
    responses: list[Upstream] = []
    processors = []
    trimmers = []
    events: list[dict] = []

    class Client:
        async def send(self, request, **kwargs):
            response = Upstream(data_by_text[request])
            responses.append(response)
            return response

    class IdentityDsp:
        def __init__(self, *args, **kwargs):
            self.metrics = PipelineMetrics()
            self.pending = []
            processors.append(self)

        def push(self, samples):
            self.metrics.audio_received_samples += len(samples)
            encoded = np.rint(samples * 32768).astype("<i2").tobytes()
            if deferred:
                self.pending.append(encoded)
                return []
            return self.emit(encoded)

        def emit(self, data):
            if not data:
                return []
            count = len(data) // 2
            self.metrics.audio_processed_samples += count
            self.metrics.audio_emitted_samples += count
            if self.metrics.first_render_started_seconds is None:
                self.metrics.first_render_started_seconds = .001
                self.metrics.first_render_finished_seconds = .002
                self.metrics.first_render_received_samples = self.metrics.audio_received_samples
            return [data]

        def finish(self):
            return self.emit(b"".join(self.pending))

    class TrackingOnset(OutputOnsetTrimmer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trimmers.append(self)

    service = SimpleNamespace(runtimes={"0" * 26: object()},
                              event=lambda runtime, name, **data: events.append({"event": name, **data}))
    monkeypatch.setattr(gateway, "_segment_upstream_request", lambda text: text)
    monkeypatch.setattr(gateway, "AcousticStreamProcessor", IdentityDsp)
    monkeypatch.setattr(gateway, "OutputOnsetTrimmer", TrackingOnset)
    monkeypatch.setattr(gateway, "ONSET_TRIM", "conservative-v1")
    monkeypatch.setattr(gateway, "PROFILE_LATENCY", True)
    monkeypatch.setattr(gateway.app.state, "client", Client(), raising=False)
    monkeypatch.setattr(gateway.app.state, "metrics", gateway.GatewayMetrics(), raising=False)
    monkeypatch.setattr(gateway.app.state, "service", service, raising=False)
    return SimpleNamespace(responses=responses, processors=processors, trimmers=trimmers,
                           events=events, metrics=gateway.app.state.metrics.values)


@pytest.mark.asyncio
async def test_adapter_opens_before_backend_eof_preserving_the_entire_suffix(monkeypatch) -> None:
    source = unit()
    state = install(monkeypatch, {"unit": source})
    stream = gateway._segment_adapter(segment(), asyncio.Event())
    first = await anext(stream)
    assert first
    assert not state.responses[0].eof
    output = first + b"".join([block async for block in stream])
    assert output == source[len(quiet(350)) * 2:]
    assert output.endswith(quiet(180).tobytes())
    assert quiet(650).tobytes() in output
    assert state.responses[0].closed
    assert state.metrics["aegis_audio_received_seconds_total"] == pytest.approx(len(source) / 48000)
    assert state.metrics["aegis_audio_emitted_seconds_total"] == pytest.approx(len(output) / 48000)
    assert state.metrics["aegis_tts_onset_removed_seconds_total"] == pytest.approx(.35)
    event = state.events[-1]
    assert event["first_raw_pcm_ms"] <= event["first_raw_voice_ms"]
    assert event["first_processed_pcm_ms"] <= event["first_output_pcm_ms"]
    assert event["audio_emitted_ms"] == pytest.approx(len(output) / 48)
    assert event["dsp_audio_emitted_ms"] == pytest.approx(len(source) / 48)
    assert event["onset_removed_ms"] == pytest.approx(350)


@pytest.mark.asyncio
async def test_each_segment_and_request_owns_its_onset_state(monkeypatch) -> None:
    sources = {"first": unit(430), "second": unit(590, 211), "other": unit(510, 251)}
    state = install(monkeypatch, sources)
    outputs = []
    for item in (segment(text="first"), segment(index=1, text="second"),
                 segment("1" * 26, text="other")):
        outputs.append(b"".join([block async for block in gateway._segment_adapter(item, asyncio.Event())]))
    assert outputs == [sources["first"][len(quiet(350)) * 2:],
                       sources["second"][len(quiet(510)) * 2:],
                       sources["other"][len(quiet(430)) * 2:]]
    assert len({id(item) for item in state.trimmers}) == 3
    assert len({id(item) for item in state.processors}) == 3
    assert all(response.closed for response in state.responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", [False, True])
async def test_cancel_between_blocks_of_one_trimmed_render_never_leaks_remainder(monkeypatch, deferred) -> None:
    state = install(monkeypatch, {"unit": unit()}, deferred=deferred)
    cancelled = asyncio.Event()
    stream = gateway._segment_adapter(segment(), cancelled)
    assert await anext(stream)
    # Opening returns retained preroll+one frame and the remaining bytes of
    # that same DSP render separately. Cancellation between them is observable.
    cancelled.set()
    assert [block async for block in stream] == []
    assert state.responses[0].closed


@pytest.mark.asyncio
async def test_cancel_during_quiet_probe_discards_only_that_request_buffer(monkeypatch) -> None:
    source = unit(590)
    state = install(monkeypatch, {"unit": source})
    cancelled = asyncio.Event()

    class CancelProbe(OutputOnsetTrimmer):
        def push(self, chunk):
            result = super().push(chunk)
            if self.metrics.decision == "pending":
                cancelled.set()
            return result

    monkeypatch.setattr(gateway, "OutputOnsetTrimmer", CancelProbe)
    assert [block async for block in gateway._segment_adapter(segment(), cancelled)] == []
    assert state.responses[0].closed
    assert state.metrics["aegis_audio_emitted_seconds_total"] == 0
    # A following request must begin with its own probe and audio.
    monkeypatch.setattr(gateway, "OutputOnsetTrimmer", OutputOnsetTrimmer)
    following = b"".join([block async for block in gateway._segment_adapter(segment("1" * 26), asyncio.Event())])
    assert following == source[len(quiet(510)) * 2:]


@pytest.mark.asyncio
async def test_disabled_variant_remains_exact_and_quiet_eof_is_preserved(monkeypatch) -> None:
    source = unit()
    state = install(monkeypatch, {"unit": source})
    monkeypatch.setattr(gateway, "ONSET_TRIM", "off")
    output = b"".join([block async for block in gateway._segment_adapter(segment(), asyncio.Event())])
    assert output == source
    assert not state.trimmers
    assert state.metrics["aegis_tts_onset_removed_seconds_total"] == 0

    silence = quiet(730).tobytes()
    state = install(monkeypatch, {"unit": silence})
    output = b"".join([block async for block in gateway._segment_adapter(segment(), asyncio.Event())])
    assert output == silence
    assert state.trimmers[0].metrics.decision == "preserved-eof"
    assert state.responses[0].eof and state.responses[0].closed
    assert state.metrics["aegis_tts_onset_removed_seconds_total"] == 0
    assert state.events[-1].get("first_raw_voice_ms") is None
    assert state.events[-1].get("first_nonsilent_pcm_ms") is None


@pytest.mark.asyncio
async def test_scheduler_keeps_segment_order_tails_and_retained_preroll(monkeypatch) -> None:
    first, second = unit(430), unit(590, 251)
    state = install(monkeypatch, {"first": first, "second": second})
    service = SpeechService(gateway._segment_adapter)
    monkeypatch.setattr(gateway.app.state, "service", service)
    request = SpeechRequest("0" * 26, "first second", "first second", 50, "enqueue",
                            "AegisVoicev1.0", True, "pcm", {})
    runtime = await service.submit(request, [segment(text="first"), segment(index=1, text="second")])
    await service.start()
    try:
        output = b"".join([block async for block in service.audio_stream(runtime)])
        expected_first, expected_second = first[len(quiet(350)) * 2:], second[len(quiet(510)) * 2:]
        assert output.startswith(expected_first)
        assert output.endswith(expected_second)
        gap = output[len(expected_first):-len(expected_second)]
        assert gap == b"\0" * len(gap)
        assert 80 <= len(gap) / 48 <= 350
        assert len(state.trimmers) == 2
        assert all(response.closed for response in state.responses)
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("path,stream,format", [
    ("/v1/audio/speech", True, "pcm"),
    ("/v1/audio/speech", False, "wav"),
    ("/v1/aegis/speech", True, "pcm"),
    ("/v1/aegis/speech", False, "wav"),
])
async def test_both_public_contracts_return_same_trimmed_pcm_and_valid_wav(monkeypatch, path, stream, format) -> None:
    source = unit()
    state = install(monkeypatch, {"unit": source})
    service = SpeechService(gateway._segment_adapter)
    monkeypatch.setattr(gateway.app.state, "service", service)

    async def submit(payload):
        request = SpeechRequest("0" * 26, "unit", "unit", 50, "enqueue", "AegisVoicev1.0",
                                payload["stream"], payload["response_format"], {})
        return await service.submit(request, [segment()])

    monkeypatch.setattr(gateway, "_submit_native", submit)
    await service.start()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://test") as client:
            response = await client.post(path, json={"input": "unit", "text": "unit", "stream": stream,
                                                     "response_format": format})
        assert response.status_code == 200
        assert response.headers["x-aegis-request-id"] == "0" * 26
        assert response.headers["x-aegis-sample-rate"] == "24000"
        assert response.headers["x-aegis-channels"] == "1"
        assert response.headers["x-aegis-voice-profile"] == gateway.VOICE_PROFILE
        expected = source[len(quiet(350)) * 2:]
        if format == "wav":
            with wave.open(io.BytesIO(response.content)) as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24000, 1, 2)
                assert wav.readframes(wav.getnframes()) == expected
        else:
            assert response.content == expected
        assert state.responses[0].closed
    finally:
        await service.stop()
