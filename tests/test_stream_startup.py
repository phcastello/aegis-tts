"""The explicit startup reservoir delays delivery, never PCM or generation."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from time import monotonic

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
from aegis_speech.domain import RequestStatus, SegmentStatus, SpeechRequest, SpeechSegment
from aegis_speech.service import SpeechService


def request(identifier: str = "0", *, stream: bool = True, response_format: str = "pcm") -> SpeechRequest:
    return SpeechRequest(identifier * 26, "texto suficiente", "texto suficiente", 50, "enqueue",
                         "AegisVoicev1.0", stream, response_format, {})


def segment(item: SpeechRequest) -> SpeechSegment:
    return SpeechSegment(item.request_id, 0, item.original_text, item.normalized_text,
                         SegmentStatus.QUEUED, 1)


async def immediate_adapter(segment, cancelled):
    for pcm in (b"\x01\x00" * 40, b"\x02\x00" * 50, b"\x03\x00" * 30):
        if cancelled.is_set():
            return
        yield pcm
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_stream_hold_is_measured_once_and_worker_produces_during_it() -> None:
    service = SpeechService(immediate_adapter, stream_startup_seconds=.08)
    item = request()
    runtime = await service.submit(item, [segment(item)])
    await service.start()
    stream = service.audio_stream(runtime)
    started = monotonic()
    first = asyncio.create_task(anext(stream))
    try:
        await asyncio.sleep(.02)
        assert not first.done()
        assert runtime.done.is_set()  # Worker has generated every PCM block.
        assert runtime.buffered_bytes == 240  # Includes the held first block.
        assert runtime.audio.qsize() == 3  # Remaining two blocks plus EOF.
        first_pcm = await first
        elapsed = monotonic() - started
        assert .065 <= elapsed < .5
        assert first_pcm == b"\x01\x00" * 40
        remainder_started = monotonic()
        remainder = [chunk async for chunk in stream]
        assert monotonic() - remainder_started < .05
        assert remainder == [b"\x02\x00" * 50, b"\x03\x00" * 30]
        assert runtime.buffered_bytes == 0
        releases = [event for event in runtime.events if event.event == "stream_startup_released"]
        assert len(releases) == 1
        assert releases[0].data["configured_ms"] == 80
        assert releases[0].data["waited_ms"] >= 65
    finally:
        await stream.aclose()
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream,delay", [(True, 0), (False, .32)])
async def test_default_and_nonstreaming_wav_do_not_wait(stream: bool, delay: float) -> None:
    service = SpeechService(immediate_adapter, stream_startup_seconds=delay)
    item = request(stream=stream, response_format="pcm" if stream else "wav")
    runtime = await service.submit(item, [segment(item)])
    await service.start()
    try:
        started = monotonic()
        output = b"".join([chunk async for chunk in service.audio_stream(runtime)])
        assert monotonic() - started < .15
        assert output == b"\x01\x00" * 40 + b"\x02\x00" * 50 + b"\x03\x00" * 30
        assert not any(event.event.startswith("stream_startup_") for event in runtime.events)
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_before_cancel", [False, True])
async def test_delete_during_hold_discards_dequeued_pcm_and_wakes_promptly(complete_before_cancel: bool) -> None:
    producing = asyncio.Event()

    async def adapter(segment, cancelled):
        yield b"\x01\x00" * 40
        producing.set()
        if not complete_before_cancel:
            await cancelled.wait()
        else:
            yield b"\x02\x00" * 40

    service = SpeechService(adapter, stream_startup_seconds=.32)
    item = request()
    runtime = await service.submit(item, [segment(item)])
    await service.start()
    stream = service.audio_stream(runtime)
    first = asyncio.create_task(anext(stream))
    try:
        await producing.wait()
        while not runtime.stream_startup_holding:
            await asyncio.sleep(0)
        if complete_before_cancel:
            await runtime.done.wait()
        started = monotonic()
        await service.cancel(item.request_id)
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(first, .1)
        assert monotonic() - started < .1
        await asyncio.wait_for(runtime.done.wait(), .1)
        assert runtime.cancel.is_set()
        assert runtime.buffered_bytes == 0
        assert item.status == (RequestStatus.COMPLETED if complete_before_cancel else RequestStatus.CANCELLED)
        assert not [event for event in runtime.events if event.event == "stream_startup_released"]
    finally:
        await stream.aclose()
        await service.stop()


@pytest.mark.asyncio
async def test_task_cancel_during_hold_cancels_generation_and_keeps_request_isolated() -> None:
    async def adapter(segment, cancelled):
        yield segment.request_id[:1].encode() * 80
        await cancelled.wait()

    service = SpeechService(adapter, stream_startup_seconds=.32)
    item = request()
    runtime = await service.submit(item, [segment(item)])
    await service.start()
    stream = service.audio_stream(runtime)
    first = asyncio.create_task(anext(stream))
    try:
        while not runtime.stream_startup_holding:
            await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await asyncio.wait_for(runtime.done.wait(), .1)
        assert item.status == RequestStatus.CANCELLED
        assert runtime.cancel.is_set()
        assert runtime.buffered_bytes == 0
        assert not runtime.stream_startup_holding
    finally:
        await stream.aclose()
        await service.stop()


@pytest.mark.asyncio
async def test_upstream_failure_during_hold_discards_already_dequeued_first_pcm() -> None:
    async def adapter(segment, cancelled):
        yield b"\x01\x00" * 40
        await asyncio.sleep(.01)
        raise RuntimeError("synthetic upstream generation failure")

    service = SpeechService(adapter, stream_startup_seconds=.05)
    item = request()
    runtime = await service.submit(item, [segment(item)])
    await service.start()
    stream = service.audio_stream(runtime)
    first = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(runtime.done.wait(), .2)
        assert item.status == RequestStatus.FAILED
        assert not runtime.cancel.is_set()  # Failure is distinct from DELETE.
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(first, .2)
        assert runtime.buffered_bytes == 0
        assert not runtime.stream_startup_holding
        assert not [event for event in runtime.events if event.event == "stream_startup_released"]
    finally:
        await stream.aclose()
        await service.stop()


@pytest.mark.asyncio
async def test_generator_close_after_first_audio_cooperatively_cancels_rest() -> None:
    async def adapter(segment, cancelled):
        yield b"\x01\x00" * 40
        await cancelled.wait()

    service = SpeechService(adapter, stream_startup_seconds=.01)
    item = request()
    runtime = await service.submit(item, [segment(item)])
    await service.start()
    stream = service.audio_stream(runtime)
    try:
        assert await anext(stream) == b"\x01\x00" * 40
        await stream.aclose()
        await asyncio.wait_for(runtime.done.wait(), .1)
        assert runtime.cancel.is_set()
        assert item.status == RequestStatus.CANCELLED
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_held_block_remains_inside_existing_audio_resource_limit() -> None:
    service = SpeechService(immediate_adapter, stream_startup_seconds=.32,
                            max_buffered_seconds=.002)
    item = request()
    runtime = await service.submit(item, [segment(item)])
    await service.start()
    try:
        output = [chunk async for chunk in service.audio_stream(runtime)]
        assert output == []
        assert runtime.cancel.is_set()
        assert item.cancel_reason == "client_stalled"
        assert runtime.buffered_bytes == 0
        assert service.metrics["client_stall_total"] == 1
    finally:
        await service.stop()


@pytest.mark.parametrize("delay", [-.001, .501, float("inf"), float("-inf"), float("nan"), None, "bad"])
def test_invalid_startup_delay_is_rejected(delay) -> None:
    with pytest.raises(ValueError, match="finite.*zero.*0.5"):
        SpeechService(immediate_adapter, stream_startup_seconds=delay)


def test_startup_delay_endpoints_are_valid() -> None:
    assert SpeechService(immediate_adapter).stream_startup_seconds == 0
    assert SpeechService(immediate_adapter, stream_startup_seconds=.5).stream_startup_seconds == .5
