"""In-memory scheduler, event ring and cooperative cancellation domain service."""
from __future__ import annotations

import asyncio
import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import AsyncIterator, Awaitable, Callable

from .domain import FINAL_REQUEST_STATUSES, RequestStatus, SegmentStatus, SpeechRequest, SpeechSegment

AudioAdapter = Callable[[SpeechSegment, asyncio.Event], AsyncIterator[bytes]]


@dataclass
class Event:
    sequence: int
    event: str
    request_id: str
    timestamp: str
    data: dict[str, object]


@dataclass
class Runtime:
    request: SpeechRequest
    segments: list[SpeechSegment]
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    audio: asyncio.Queue[bytes | None] = field(default_factory=lambda: asyncio.Queue(maxsize=32))
    buffered_bytes: int = 0
    events: deque[Event] = field(default_factory=lambda: deque(maxlen=256))
    sequence: int = 0
    created_monotonic: float = field(default_factory=monotonic)
    queued_monotonic: float = field(default_factory=monotonic)
    stream_startup_holding: bool = False


class QueueFullError(RuntimeError): pass
class BusyError(RuntimeError): pass


class InterruptionPlanner:
    def __init__(self, allow_replace_all: bool = False) -> None:
        self.allow_replace_all = allow_replace_all

    def apply_policy(self, incoming: SpeechRequest, active: SpeechRequest | None, queued: list[SpeechRequest]) -> tuple[list[str], list[str]]:
        all_requests = ([active] if active else []) + queued
        if incoming.interrupt_policy == "reject_if_busy" and all_requests:
            raise BusyError("request queue is busy")
        if incoming.interrupt_policy == "replace_all":
            if not self.allow_replace_all:
                raise PermissionError("replace_all is disabled")
            return [r.request_id for r in all_requests], []
        if incoming.interrupt_policy == "replace_lower_priority":
            cancelled = [r.request_id for r in all_requests if r.priority < incoming.priority]
            return cancelled, [r.request_id for r in all_requests if r.request_id not in cancelled]
        return [], [r.request_id for r in all_requests]


class SpeechService:
    """A single-worker, bounded scheduler. Aging bonus is floor(wait/interval), capped."""
    def __init__(self, adapter: AudioAdapter, *, max_requests: int = 32, max_segments: int = 128, max_characters: int = 50000, aging_interval: int = 30, aging_max_bonus: int = 10, max_buffered_seconds: float = 4.0, sample_rate: int = 24000, allow_replace_all: bool = False, stream_startup_seconds: float = 0.0) -> None:
        try:
            startup_seconds = float(stream_startup_seconds)
        except (ValueError, TypeError) as exc:
            raise ValueError("stream startup delay must be finite and between zero and 0.5 seconds") from exc
        if not math.isfinite(startup_seconds) or not 0 <= startup_seconds <= .5:
            raise ValueError("stream startup delay must be finite and between zero and 0.5 seconds")
        self.adapter = adapter
        self.stream_startup_seconds = startup_seconds
        self.max_requests, self.max_segments, self.max_characters = max_requests, max_segments, max_characters
        self.aging_interval, self.aging_max_bonus = aging_interval, aging_max_bonus
        self.max_buffered_bytes = int(max_buffered_seconds * sample_rate * 2)
        self.sample_rate = sample_rate
        self.join_min_samples, self.join_default_samples, self.join_max_samples = int(sample_rate * .080), int(sample_rate * .160), int(sample_rate * .350)
        self.runtimes: dict[str, Runtime] = {}
        self._heap: list[tuple[int, int, str, int]] = []
        self._counter = itertools.count()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self.active_id: str | None = None
        self.planner = InterruptionPlanner(allow_replace_all)
        self.metrics: dict[str, float] = {"requests_total": 0, "completed_total": 0, "cancelled_total": 0, "failed_total": 0, "segments_generated_total": 0, "client_stall_total": 0, "queue_wait_seconds": 0}

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="aegis-speech-worker")

    async def stop(self) -> None:
        self._stopping = True
        for request_id in list(self.runtimes):
            await self.cancel(request_id, "shutdown")
        self._wake.set()
        if self._task:
            await self._task

    def event(self, runtime: Runtime, name: str, **data: object) -> None:
        runtime.sequence += 1
        runtime.events.append(Event(runtime.sequence, name, runtime.request.request_id, datetime.now(timezone.utc).isoformat(), {"elapsed_ms": (monotonic() - runtime.created_monotonic) * 1000, **data}))

    async def submit(self, request: SpeechRequest, segments: list[SpeechSegment]) -> Runtime:
        if request.request_id in self.runtimes:
            raise ValueError("request_id already exists")
        unfinished = [r for r in self.runtimes.values() if r.request.status not in FINAL_REQUEST_STATUSES]
        if len(unfinished) >= self.max_requests or sum(len(r.segments) for r in unfinished) + len(segments) > self.max_segments or sum(len(r.request.original_text) for r in unfinished) + len(request.original_text) > self.max_characters:
            raise QueueFullError("queue capacity exceeded")
        active = self.runtimes[self.active_id].request if self.active_id else None
        queued = [r.request for r in unfinished if r.request.request_id != self.active_id]
        cancelled, _ = self.planner.apply_policy(request, active, queued)
        for request_id in cancelled:
            await self.cancel(request_id, "policy")
        runtime = Runtime(request, segments, created_monotonic=request.created_monotonic)
        self.runtimes[request.request_id] = runtime
        self.event(runtime, "accepted")
        request.transition(RequestStatus.NORMALIZING)
        self.event(runtime, "normalization_completed", segment_count=len(segments))
        request.transition(RequestStatus.QUEUED)
        self.event(runtime, "queued", position=len(self._heap) + 1)
        self._push(request.request_id, 0)
        self.metrics["requests_total"] += 1
        self._wake.set()
        return runtime

    def _push(self, request_id: str, segment_index: int) -> None:
        runtime = self.runtimes[request_id]
        runtime.queued_monotonic = monotonic()
        self.event(runtime, "segment_queued", segment_index=segment_index)
        # The heap base is recalculated by worker for aging; sequence preserves FIFO.
        heapq.heappush(self._heap, (-runtime.request.priority, next(self._counter), request_id, segment_index))

    def queue_position(self, request_id: str) -> int:
        ids = [entry[2] for entry in sorted(self._heap)]
        return (ids.index(request_id) + 1) if request_id in ids else 0

    async def cancel(self, request_id: str, reason: str = "client_request") -> Runtime | None:
        runtime = self.runtimes.get(request_id)
        if not runtime or runtime.request.status in FINAL_REQUEST_STATUSES:
            if runtime and runtime.stream_startup_holding and not runtime.cancel.is_set():
                # Synthesis can finish while its first PCM is held. Preserve
                # that historical completion status, but DELETE must still
                # discard audio that has never reached the client.
                runtime.cancel.set()
                runtime.request.cancel_reason = reason
                self.event(runtime, "stream_startup_cancelled", reason=reason)
                await self._close_audio(runtime, discard=True)
            return runtime
        if runtime.request.status != RequestStatus.CANCELLING:
            runtime.request.transition(RequestStatus.CANCELLING)
            runtime.request.cancel_reason = reason
            runtime.cancel.set()
        if self.active_id != request_id:
            await self._finish_cancel(runtime)
        return runtime

    async def _finish_cancel(self, runtime: Runtime) -> None:
        if runtime.request.status == RequestStatus.CANCELLING:
            runtime.request.transition(RequestStatus.CANCELLED)
            self.metrics["cancelled_total"] += 1
            self.event(runtime, "cancelled", reason=runtime.request.cancel_reason or "client_request")
            runtime.done.set()
            await self._close_audio(runtime, discard=True)

    async def _close_audio(self, runtime: Runtime, *, discard: bool = False) -> None:
        if discard:
            while not runtime.audio.empty():
                try: runtime.audio.get_nowait()
                except asyncio.QueueEmpty: break
            runtime.buffered_bytes = 0
            try: runtime.audio.put_nowait(None)
            except asyncio.QueueFull: pass
            return
        # Successful requests must preserve every queued PCM block.  Draining
        # here races a non-streaming HTTP consumer and truncates the tail.
        await runtime.audio.put(None)

    async def _run(self) -> None:
        while True:
            entry = self._next()
            if entry is None:
                if self._stopping:
                    return
                self._wake.clear()
                await self._wake.wait()
                if self._stopping: return
                continue
            _, _, request_id, index = entry
            runtime = self.runtimes.get(request_id)
            if not runtime or runtime.cancel.is_set() or runtime.request.status in FINAL_REQUEST_STATUSES:
                continue
            self.active_id = request_id
            await self._generate(runtime, index)
            self.active_id = None

    def _next(self) -> tuple[int, int, str, int] | None:
        # Aging is bounded: effective priority = priority + min(10, floor(wait/30)).
        candidates = []
        while self._heap:
            candidates.append(heapq.heappop(self._heap))
        if not candidates: return None
        now = datetime.now(timezone.utc)
        candidates.sort(key=lambda item: (-(self.runtimes[item[2]].request.priority + min(self.aging_max_bonus, int((now - (self.runtimes[item[2]].request.queued_at or now)).total_seconds() // self.aging_interval))), item[1], item[3]))
        chosen, rest = candidates[0], candidates[1:]
        self._heap.extend(rest); heapq.heapify(self._heap)
        return chosen

    async def _generate(self, runtime: Runtime, index: int) -> None:
        request, segment = runtime.request, runtime.segments[index]
        try:
            request.transition(RequestStatus.GENERATING)
            segment.status, segment.started_at = SegmentStatus.GENERATING, datetime.now(timezone.utc)
            queue_wait = monotonic() - runtime.queued_monotonic
            self.metrics["queue_wait_seconds"] += queue_wait
            self.event(runtime, "generation_started", segment_index=index, queue_wait_ms=queue_wait * 1000)
            first = True
            tail = b""
            async for pcm in self.adapter(segment, runtime.cancel):
                if runtime.cancel.is_set(): break
                if first:
                    first = False
                    request.first_audio_at = datetime.now(timezone.utc)
                    request.transition(RequestStatus.STREAMING)
                    self.event(runtime, "audio_started", segment_index=index)
                await asyncio.wait_for(runtime.audio.put(pcm), timeout=10)
                runtime.buffered_bytes += len(pcm)
                tail = (tail + pcm)[-self.join_max_samples * 2:]
                if runtime.buffered_bytes > self.max_buffered_bytes:
                    raise TimeoutError("pcm buffer limit exceeded")
            if runtime.cancel.is_set():
                await self._finish_cancel(runtime); return
            segment.status, segment.completed_at = SegmentStatus.COMPLETED, datetime.now(timezone.utc)
            self.metrics["segments_generated_total"] += 1
            self.event(runtime, "segment_completed", segment_index=index)
            if index + 1 < len(runtime.segments):
                # The acoustic adapter preserves the natural tail.  Only fill
                # enough measured digital silence to reach the configured
                # default (bounded by the 80–350 ms join contract).
                quiet = self._trailing_silence_samples(tail)
                fill = min(self.join_max_samples, max(self.join_min_samples, self.join_default_samples - quiet))
                if fill:
                    gap = b"\x00\x00" * fill
                    await asyncio.wait_for(runtime.audio.put(gap), timeout=10)
                    runtime.buffered_bytes += len(gap)
                request.transition(RequestStatus.QUEUED)
                self._push(request.request_id, index + 1); self._wake.set()
            else:
                request.transition(RequestStatus.COMPLETED)
                self.metrics["completed_total"] += 1
                self.event(runtime, "completed")
                runtime.done.set(); await self._close_audio(runtime)
        except (TimeoutError, asyncio.TimeoutError):
            self.metrics["client_stall_total"] += 1
            await self.cancel(request.request_id, "client_stalled")
            await self._finish_cancel(runtime)
        except Exception:
            if runtime.cancel.is_set():
                await self._finish_cancel(runtime)
            else:
                request.error_code = "UPSTREAM_GENERATION_FAILED"
                request.transition(RequestStatus.FAILED)
                self.metrics["failed_total"] += 1
                self.event(runtime, "failed", error_code=request.error_code)
                runtime.done.set(); await self._close_audio(runtime, discard=True)

    async def _wait_stream_startup(self, runtime: Runtime) -> bool:
        runtime.stream_startup_holding = True
        started = monotonic()
        self.event(runtime, "stream_startup_buffering", configured_ms=self.stream_startup_seconds * 1000)
        try:
            await asyncio.wait_for(runtime.cancel.wait(), timeout=self.stream_startup_seconds)
        except asyncio.TimeoutError:
            pass
        if runtime.cancel.is_set():
            return False
        runtime.stream_startup_holding = False
        self.event(runtime, "stream_startup_released", configured_ms=self.stream_startup_seconds * 1000,
                   waited_ms=(monotonic() - started) * 1000)
        return True

    async def audio_stream(self, runtime: Runtime) -> AsyncIterator[bytes]:
        first = True
        try:
            while True:
                chunk = await runtime.audio.get()
                if chunk is None: break
                if first and runtime.request.stream and self.stream_startup_seconds:
                    # Keep the one dequeued block in buffered_bytes during
                    # this explicit startup delay. The producer keeps using
                    # the existing bounded queue and four-second audio cap;
                    # there is no second growing prebuffer or waveform edit.
                    if not await self._wait_stream_startup(runtime):
                        return
                first = False
                if runtime.cancel.is_set():
                    return
                runtime.buffered_bytes = max(0, runtime.buffered_bytes - len(chunk))
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            await self.cancel(runtime.request.request_id, "client_disconnected")
            raise
        finally:
            runtime.stream_startup_holding = False

    @staticmethod
    def _trailing_silence_samples(pcm: bytes, threshold: int = 100) -> int:
        """Count PCM16 tail samples without numpy or endian-dependent casts."""
        count = 0
        for offset in range(len(pcm) - 2, -1, -2):
            value = int.from_bytes(pcm[offset:offset + 2], "little", signed=True)
            if abs(value) > threshold:
                break
            count += 1
        return count
