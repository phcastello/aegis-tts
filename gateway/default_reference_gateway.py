"""Private-reference AegisVoicev1.0 gateway for vLLM-Omni Qwen Base.

The only public synthesis route remains OpenAI-compatible
``POST /v1/audio/speech``.  Reference material and the Q00 sampling controls
are server-owned invariants, never client-selectable request options.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import io
import wave
import secrets
import re
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response, StreamingResponse

from voice_pipeline import AcousticStreamProcessor, Pcm16Reassembler, PcmProtocolError
from aegis_speech.domain import RequestStatus, SegmentStatus, SpeechRequest, SpeechSegment
from aegis_speech.lexicon import PronunciationLexicon
from aegis_speech.normalization import SpeechTextNormalizer
from aegis_speech.segmentation import SpeechSegmenter
from aegis_speech.service import BusyError, QueueFullError, Runtime, SpeechService


BACKEND_URL = os.environ.get("AEGIS_VLLM_BACKEND_URL", "http://127.0.0.1:8091")  # operational endpoint only
PROTECTED_ROOT = Path("/protected")
DEFAULT_AUDIO_PATH = PROTECTED_ROOT / "reference/Aninha_AegisVoice4.wav"
DEFAULT_TEXT_PATH = PROTECTED_ROOT / "reference/Aninha_AegisVoice4.txt"
MANIFEST_PATH = PROTECTED_ROOT / "provenance/SHA256SUMS"
VOICE_CONFIG_PATH = PROTECTED_ROOT / "config/qwen-selected-p07-m02-v1.0.yaml"
Q00_CONFIG_PATH = PROTECTED_ROOT / "config/q00-vllm-deploy-exact.yaml"
PROFILE_CONFIG_PATH = PROTECTED_ROOT / "config/profile.yaml"
LEXICON_PATH = Path(os.environ.get("AEGIS_LEXICON_PATH", str(PROTECTED_ROOT / "pronunciation/pt-BR.yaml")))
DEFAULT_MODEL = "/models/Qwen3-TTS-12Hz-0.6B-Base"
SAMPLE_RATE = 24000
PROCESSING_MODE = "streaming"
VOICE_PROFILE = "AegisVoicev1.0-Q00-W04-P07-M02"
NATIVE_PROFILE = "AegisVoicev1.0"
INITIAL_SEGMENT = os.environ.get("AEGIS_TTS_INITIAL_SEGMENT", "stable")
PROFILE_LATENCY = os.environ.get("AEGIS_TTS_PROFILE_LATENCY") == "1"
SOURCE_COMMIT = os.environ.get("AEGIS_TTS_SOURCE_COMMIT", "unknown")
REQUEST_ID = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")

Q00_PARAMS: dict[str, Any] = {
    "temperature": 0.05,
    "top_p": 0.3,
    "top_k": 1,
    "repetition_penalty": 1.15,
    "seed": 40733133,
    "min_tokens": 0,
    "max_tokens": 1024,
    # This preserves the exact Q00 prompt construction in vLLM-Omni.  It is
    # independent of the HTTP ``stream`` flag and does not disable streaming.
    "non_streaming_mode": True,
}


class GatewayMetrics:
    """Small dependency-free metrics registry; values never contain request text."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {
            "aegis_active_syntheses": 0,
            "aegis_synthesis_cancellations_total": 0,
            "aegis_pcm_invalid_residual_bytes_total": 0,
            "aegis_audio_received_seconds_total": 0,
            "aegis_audio_processed_seconds_total": 0,
            "aegis_audio_emitted_seconds_total": 0,
            "aegis_segments_natural_total": 0,
            "aegis_segments_hard_seam_total": 0,
            "aegis_ttfa_processed_seconds": 0,
            "aegis_dsp_lag_seconds": 0,
            "aegis_tts_requests_total": 0,
            "aegis_tts_requests_active": 0,
            "aegis_tts_requests_queued": 0,
            "aegis_tts_requests_completed_total": 0,
            "aegis_tts_requests_cancelled_total": 0,
            "aegis_tts_requests_failed_total": 0,
            "aegis_tts_segments_queued": 0,
            "aegis_tts_segments_generated_total": 0,
            "aegis_tts_pcm_buffered_seconds": 0,
            "aegis_tts_client_stall_total": 0,
            "aegis_tts_upstream_errors_total": 0,
            "aegis_tts_queue_wait_seconds": 0,
            "aegis_tts_normalization_seconds": 0,
            "aegis_tts_segmentation_seconds": 0,
            "aegis_tts_generation_seconds": 0,
            "aegis_tts_dsp_seconds": 0,
            "aegis_tts_time_to_first_audio_seconds": 0,
            "aegis_tts_end_to_end_seconds": 0,
            "aegis_tts_lexicon_version_info": 1,
            "aegis_tts_voice_profile_info": 1,
            "aegis_tts_first_raw_pcm_seconds": 0,
            "aegis_tts_first_processed_pcm_seconds": 0,
            "aegis_tts_first_nonsilent_pcm_seconds": 0,
            "aegis_tts_first_dsp_input_audio_seconds": 0,
        }

    def render(self) -> str:
        return "".join(f"{name} {value}\n" for name, value in sorted(self.values.items()))


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        checksum, relative = line.split(maxsplit=1)
        entries[relative.strip().removeprefix("./")] = checksum
    return entries


def _verify_protected_assets() -> None:
    """Verify every runtime-critical protected file before accepting traffic."""
    entries = _manifest_entries(MANIFEST_PATH)
    for target in (DEFAULT_AUDIO_PATH, DEFAULT_TEXT_PATH, VOICE_CONFIG_PATH, Q00_CONFIG_PATH, PROFILE_CONFIG_PATH, LEXICON_PATH):
        try:
            relative = target.relative_to(PROTECTED_ROOT).as_posix()
        except ValueError as exc:
            raise RuntimeError("protected runtime asset is outside the protected mount") from exc
        expected = entries.get(relative)
        if not expected or not target.is_file() or _digest(target) != expected:
            raise RuntimeError("protected voice asset integrity check failed")


def _load_default_reference() -> tuple[str, str]:
    """Load once at startup; reference data is never logged or returned."""
    try:
        audio = DEFAULT_AUDIO_PATH.read_bytes()
        transcript = DEFAULT_TEXT_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError("protected default reference is unavailable") from exc
    if not audio or not transcript:
        raise RuntimeError("protected default reference is empty")
    return "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii"), transcript


def _check_dsp_imports() -> None:
    try:
        import numpy  # noqa: F401
        import pyworld  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Aegis DSP runtime is unavailable") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PROCESSING_MODE not in {"streaming", "offline", "bypass"}:
        raise RuntimeError("AEGIS_VOICE_PROCESSING_MODE must be streaming, offline, or bypass")
    _verify_protected_assets()
    _check_dsp_imports()
    AcousticStreamProcessor(SAMPLE_RATE, initial_segment=INITIAL_SEGMENT)
    app.state.default_audio, app.state.default_text = _load_default_reference()
    app.state.client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=httpx.Timeout(900.0, connect=5.0))
    app.state.synthesis_slot = asyncio.Semaphore(1)
    app.state.metrics = GatewayMetrics()
    lexicon = PronunciationLexicon(LEXICON_PATH)
    app.state.normalizer = SpeechTextNormalizer(lexicon.apply)
    app.state.segmenter = SpeechSegmenter()
    app.state.service = SpeechService(_segment_adapter, allow_replace_all=os.environ.get("AEGIS_ALLOW_REPLACE_ALL") == "1")
    await app.state.service.start()
    app.state.warmed = False
    app.state.warm_lock = asyncio.Lock()
    try:
        yield
    finally:
        await app.state.service.stop()
        await app.state.client.aclose()


app = FastAPI(title="Aegis TTS vLLM gateway", docs_url=None, redoc_url=None, lifespan=lifespan)


WEB_PLAYER = """<!doctype html>
<html lang="pt-BR">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aegis Voice</title>
<style>
  :root { color-scheme: dark; font-family: system-ui, sans-serif; }
  body { max-width: 42rem; margin: 4rem auto; padding: 0 1rem; background: #101318; color: #e7edf4; }
  textarea { box-sizing: border-box; width: 100%; min-height: 9rem; padding: .8rem; border: 1px solid #3c4756; border-radius: .4rem; background: #181d25; color: inherit; font: inherit; }
  button { margin-top: .8rem; margin-right: .5rem; padding: .6rem 1rem; border: 0; border-radius: .4rem; background: #80c7ff; color: #07111c; font-weight: 650; cursor: pointer; }
  button[disabled] { opacity: .5; cursor: wait; }
  #stop { background: #d9a4a4; }
  #status { min-height: 1.4em; margin-top: 1rem; color: #aebdcd; }
</style>
<main>
  <h1>Aegis Voice</h1>
  <textarea id="text" autofocus>Teste breve de streaming da voz Aegis.</textarea>
  <div><button id="send">Falar</button><button id="stop" disabled>Parar</button></div>
  <p id="status" role="status">Pronto.</p>
  <p id="metric" aria-live="polite"></p>
</main>
<script>
(() => {
  const sampleRate = 24000;
  const text = document.querySelector('#text');
  const send = document.querySelector('#send');
  const stop = document.querySelector('#stop');
  const status = document.querySelector('#status');
  const metric = document.querySelector('#metric');
  let context;
  let controller;
  let nextStart = 0;
  let remainder = new Uint8Array(0);
  let clickStartedAt = 0;
  let firstAudioScheduled = false;
  const sources = new Set();

  function setStatus(message) { status.textContent = message; }
  function stopAudio() {
    controller?.abort();
    controller = undefined;
    for (const source of sources) { try { source.stop(); } catch (_) {} }
    sources.clear();
    remainder = new Uint8Array(0);
    stop.disabled = true;
    send.disabled = false;
  }
  function queuePcm(chunk) {
    const combined = new Uint8Array(remainder.length + chunk.length);
    combined.set(remainder);
    combined.set(chunk, remainder.length);
    const usable = combined.length - (combined.length % 2);
    remainder = combined.slice(usable);
    if (!usable) return;
    const pcm = new DataView(combined.buffer, combined.byteOffset, usable);
    const buffer = context.createBuffer(1, usable / 2, sampleRate);
    const output = buffer.getChannelData(0);
    for (let index = 0; index < output.length; index += 1) {
      output[index] = pcm.getInt16(index * 2, true) / 32768;
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.onended = () => sources.delete(source);
    nextStart = Math.max(nextStart, context.currentTime + 0.05);
    const startAt = nextStart;
    source.start(startAt);
    nextStart += buffer.duration;
    sources.add(source);
    if (!firstAudioScheduled) {
      firstAudioScheduled = true;
      // getOutputTimestamp maps the Web Audio clock to the output device's
      // performance clock. The fallback remains a scheduling estimate.
      const output = context.getOutputTimestamp?.();
      const playedAt = output
        ? output.performanceTime + (startAt - output.contextTime) * 1000
        : performance.now() + (startAt - context.currentTime) * 1000;
      const endToEndMs = Math.max(0, playedAt - clickStartedAt);
      window.setTimeout(() => {
        metric.textContent = `Início ponta a ponta: ${Math.round(endToEndMs)} ms`;
        setStatus('Falando…');
      }, Math.max(0, playedAt - performance.now()));
    }
  }
  async function speak() {
    const input = text.value.trim();
    if (!input) { setStatus('Digite um texto.'); return; }
    clickStartedAt = performance.now();
    firstAudioScheduled = false;
    metric.textContent = '';
    stopAudio();
    context ??= new AudioContext({ sampleRate });
    await context.resume(); // The click is the browser's required audio gesture.
    controller = new AbortController();
    nextStart = context.currentTime + 0.05;
    send.disabled = true;
    stop.disabled = false;
    setStatus('Gerando…');
    try {
      const response = await fetch('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({ input, stream: true, stream_format: 'audio', response_format: 'pcm' }),
      });
      if (!response.ok || !response.body) {
        throw new Error(`Erro HTTP ${response.status}`);
      }
      const profile = response.headers.get('X-Aegis-Voice-Profile');
      setStatus(profile ? `Recebendo áudio (${profile})…` : 'Recebendo áudio…');
      const reader = response.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        queuePcm(value);
      }
      if (remainder.length) throw new Error('Fluxo PCM inválido.');
      const wait = Math.max(0, (nextStart - context.currentTime) * 1000);
      window.setTimeout(() => { if (!controller) return; stopAudio(); setStatus('Concluído.'); }, wait + 25);
    } catch (error) {
      if (error.name !== 'AbortError') setStatus(error.message || 'Falha ao gerar áudio.');
      stopAudio();
    }
  }
  send.addEventListener('click', speak);
  stop.addEventListener('click', () => { stopAudio(); setStatus('Interrompido.'); });
})();
</script>
</html>"""


def _apply_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    """Make identity and Q00 deterministic, ignoring public voice controls."""
    request = dict(payload)
    # This direct assignment is intentionally kept explicit: every Base call
    # carries the approved ICL pair, including requests with hostile fields.
    request["ref_audio"] = app.state.default_audio
    request["ref_text"] = app.state.default_text
    request.pop("voice", None)
    request["model"] = DEFAULT_MODEL
    request["task_type"] = "Base"
    request["x_vector_only_mode"] = False
    request["language"] = "Portuguese"
    request.update(Q00_PARAMS)
    if request.get("stream"):
        request["stream_format"] = "audio"
        request["response_format"] = "pcm"
    else:
        # The gateway must receive raw samples to render the offline golden
        # path; the caller's requested container is restored at the edge.
        request["response_format"] = "pcm" if PROCESSING_MODE != "bypass" else request.get("response_format", "wav")
    return request


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    allowed = ("content-type", "content-disposition")
    headers = {name: upstream.headers[name] for name in allowed if name in upstream.headers}
    headers["X-Aegis-Voice-Profile"] = VOICE_PROFILE
    return headers


async def _backend_healthy() -> bool:
    try:
        response = await app.state.client.get("/health")
    except httpx.HTTPError:
        return False
    return response.is_success


async def _warmup() -> bool:
    """Warm Q00/reference internals without exposing an audio response."""
    if app.state.warmed:
        return True
    async with app.state.warm_lock:
        if app.state.warmed:
            return True
        request = _apply_defaults({"input": "Aegis.", "response_format": "pcm", "stream": False})
        await _acquire_slot()
        try:
            response = await app.state.client.post("/v1/audio/speech", json=request)
        except httpx.HTTPError:
            return False
        finally:
            _release_slot()
        app.state.warmed = response.is_success
        return app.state.warmed


async def _acquire_slot() -> None:
    # No await occurs between the check and acquire when the semaphore has a
    # permit, so a second request is deterministically rejected with 429.
    if app.state.synthesis_slot.locked():
        raise HTTPException(status_code=429, detail="Aegis synthesis is busy", headers={"Retry-After": "2"})
    await app.state.synthesis_slot.acquire()
    app.state.metrics.values["aegis_active_syntheses"] += 1


def _release_slot() -> None:
    app.state.synthesis_slot.release()
    app.state.metrics.values["aegis_active_syntheses"] -= 1


def _pcm_wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return output.getvalue()


def _ulid() -> str:
    """Monotonic-sortable, URL-safe ULID; avoids accepting arbitrary IDs."""
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    value = int(__import__("time").time() * 1000).to_bytes(6, "big") + secrets.token_bytes(10)
    number = int.from_bytes(value, "big")
    chars: list[str] = []
    for _ in range(26):
        chars.append(alphabet[number & 31])
        number >>= 5
    return "".join(reversed(chars))


def _native_error(status: int, code: str, request_id: str | None = None, detail: str | None = None) -> HTTPException:
    payload: dict[str, str] = {"code": code}
    if request_id:
        payload["request_id"] = request_id
    if detail:
        payload["message"] = detail
    return HTTPException(status_code=status, detail=payload)


def _native_request(payload: dict[str, Any]) -> SpeechRequest:
    allowed = {"request_id", "text", "priority", "interrupt_policy", "stream", "response_format", "voice_profile", "metadata"}
    unknown = set(payload) - allowed
    if unknown:
        raise _native_error(422, "INVALID_REQUEST", detail="unknown request fields")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise _native_error(422, "INVALID_REQUEST", detail="text is required")
    if len(text) > 12000:
        raise _native_error(413, "TEXT_TOO_LONG")
    request_id = payload.get("request_id") or _ulid()
    if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
        raise _native_error(422, "INVALID_REQUEST", detail="request_id must be a ULID")
    profile = payload.get("voice_profile", NATIVE_PROFILE)
    if profile != NATIVE_PROFILE:
        raise _native_error(422, "UNSUPPORTED_PROFILE", request_id)
    priority = payload.get("priority", 50)
    if not isinstance(priority, int) or isinstance(priority, bool) or not 0 <= priority <= 100:
        raise _native_error(422, "INVALID_REQUEST", request_id, "priority must be 0..100")
    policy = payload.get("interrupt_policy", "enqueue")
    if policy not in {"enqueue", "replace_lower_priority", "replace_all", "reject_if_busy"}:
        raise _native_error(422, "INVALID_REQUEST", request_id, "unsupported interrupt_policy")
    stream = payload.get("stream", True)
    response_format = payload.get("response_format", "pcm" if stream else "wav")
    if not isinstance(stream, bool) or response_format not in {"pcm", "wav"}:
        raise _native_error(422, "UNSUPPORTED_FORMAT", request_id)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict) or set(metadata) - {"conversation_id", "source", "trace_id"} or any(not isinstance(v, str) or len(v) > 256 for v in metadata.values()):
        raise _native_error(422, "INVALID_REQUEST", request_id, "invalid metadata")
    return SpeechRequest(request_id, text, None, priority, policy, NATIVE_PROFILE, stream, response_format, dict(metadata))


async def _dsp_call(function: Any, *args: Any) -> Any:
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # A running thread cannot be cancelled. Join this single owned DSP
        # operation before releasing the adapter; its state is never reused.
        await task
        raise


async def _segment_adapter(segment: SpeechSegment, cancelled: asyncio.Event) -> AsyncIterator[bytes]:
    """One Qwen request per text segment; acoustic settings remain server-owned."""
    started = perf_counter()
    stamps: dict[str, float] = {}
    payload = _apply_defaults({"input": segment.normalized_text, "stream": True, "response_format": "pcm"})
    try:
        upstream = await app.state.client.send(app.state.client.build_request("POST", "/v1/audio/speech", json=payload), stream=True)
    except httpx.HTTPError as exc:
        raise RuntimeError("upstream unavailable") from exc
    if upstream.is_error:
        await upstream.aclose()
        raise RuntimeError("upstream generation failed")
    stamps["upstream_headers_ms"] = (perf_counter() - started) * 1000
    reassembler, processor = Pcm16Reassembler(), AcousticStreamProcessor(SAMPLE_RATE, initial_segment=INITIAL_SEGMENT)
    processor_offset_ms = (perf_counter() - started) * 1000
    first_raw = first_processed = first_nonsilent = True
    dsp_seconds = 0.0
    received_samples = 0

    def observe(rendered: bytes) -> None:
        nonlocal first_processed, first_nonsilent
        elapsed = perf_counter() - started
        if first_processed:
            first_processed = False
            stamps["first_processed_pcm_ms"] = elapsed * 1000
            app.state.metrics.values["aegis_tts_first_processed_pcm_seconds"] = elapsed
        if first_nonsilent and np.any(np.abs(np.frombuffer(rendered, dtype="<i2").astype(np.int32)) > 100):
            first_nonsilent = False
            stamps["first_nonsilent_pcm_ms"] = elapsed * 1000
            app.state.metrics.values["aegis_tts_first_nonsilent_pcm_seconds"] = elapsed

    try:
        async for chunk in upstream.aiter_bytes():
            if cancelled.is_set():
                return
            samples = reassembler.feed(chunk)
            received_samples += len(samples)
            if len(samples) and first_raw:
                first_raw = False
                stamps["first_raw_pcm_ms"] = (perf_counter() - started) * 1000
                app.state.metrics.values["aegis_tts_first_raw_pcm_seconds"] = stamps["first_raw_pcm_ms"] / 1000
            # One processor belongs to one sequential adapter. WORLD and the
            # stateful mastering chain run on a worker to reduce event-loop
            # stalls (C extensions can still hold the GIL). Bounded
            # pieces also avoid processing an arbitrarily large transport
            # chunk in full before exposing its first useful output.
            for offset in range(0, len(samples), SAMPLE_RATE // 4):
                if cancelled.is_set():
                    return
                before = perf_counter()
                rendered_chunks = await _dsp_call(processor.push, samples[offset:offset + SAMPLE_RATE // 4])
                dsp_seconds += perf_counter() - before
                for rendered in rendered_chunks:
                    if rendered and not cancelled.is_set():
                        observe(rendered)
                        yield rendered
        if cancelled.is_set():
            return
        reassembler.finish()
        stamps["upstream_eof_ms"] = (perf_counter() - started) * 1000
        before = perf_counter()
        final_chunks = await _dsp_call(processor.finish)
        dsp_seconds += perf_counter() - before
        for rendered in final_chunks:
            if rendered and not cancelled.is_set():
                observe(rendered)
                yield rendered
    finally:
        await upstream.aclose()
        values, metrics = app.state.metrics.values, processor.metrics
        values["aegis_audio_received_seconds_total"] += received_samples / SAMPLE_RATE
        values["aegis_audio_processed_seconds_total"] += metrics.audio_processed_samples / SAMPLE_RATE
        values["aegis_audio_emitted_seconds_total"] += metrics.audio_emitted_samples / SAMPLE_RATE
        values["aegis_segments_natural_total"] += metrics.natural_segments
        values["aegis_segments_hard_seam_total"] += metrics.hard_seams
        values["aegis_tts_dsp_seconds"] += dsp_seconds
        values["aegis_tts_generation_seconds"] += perf_counter() - started
        values["aegis_tts_first_dsp_input_audio_seconds"] = metrics.first_render_received_samples / SAMPLE_RATE
        if PROFILE_LATENCY:
            # A single bounded aggregate event, correlated to the existing
            # request ID. No text, voice reference, credential or raw PCM.
            runtime = app.state.service.runtimes.get(segment.request_id)
            if runtime:
                app.state.service.event(runtime, "synthesis_profiled", segment_index=segment.segment_index,
                    initial_segment=INITIAL_SEGMENT, cancelled=cancelled.is_set(), **stamps,
                    dsp_ms=dsp_seconds * 1000, world_ms=metrics.world_seconds * 1000,
                    first_dsp_input_audio_ms=metrics.first_render_received_samples / SAMPLE_RATE * 1000,
                    first_dsp_start_ms=processor_offset_ms + metrics.first_render_started_seconds * 1000 if metrics.first_render_started_seconds is not None else None,
                    first_dsp_end_ms=processor_offset_ms + metrics.first_render_finished_seconds * 1000 if metrics.first_render_finished_seconds is not None else None,
                    audio_received_ms=received_samples / SAMPLE_RATE * 1000,
                    audio_emitted_ms=metrics.audio_emitted_samples / SAMPLE_RATE * 1000)


async def _processed_stream(upstream: httpx.Response, mode: str) -> AsyncIterator[bytes]:
    reassembler = Pcm16Reassembler()
    processor = AcousticStreamProcessor(SAMPLE_RATE)
    offline_samples: list[np.ndarray] = []
    started = perf_counter()
    first_audio = True
    try:
        async for chunk in upstream.aiter_bytes():
            samples = reassembler.feed(chunk)
            if mode == "bypass":
                yield chunk
                continue
            if mode == "offline":
                # The reassembler is still needed so final odd-byte validation
                # remains identical to the streaming transport contract.
                offline_samples.append(samples)
                continue
            before = perf_counter()
            for rendered in processor.push(samples):
                if rendered:
                    if first_audio:
                        app.state.metrics.values["aegis_ttfa_processed_seconds"] = perf_counter() - started
                        first_audio = False
                    yield rendered
            processing_seconds = perf_counter() - before
            app.state.metrics.values["aegis_dsp_lag_seconds"] = max(0.0, processing_seconds - len(samples) / SAMPLE_RATE)
            if processing_seconds > 3.0:
                raise RuntimeError("Aegis DSP lag exceeded three seconds")
            if processor.buffered_seconds > 4.0:
                raise RuntimeError("Aegis DSP backlog exceeded four seconds")
        reassembler.finish()
        if mode != "bypass":
            if mode == "offline":
                # Offline's golden render intentionally has one full analysis.
                yield processor.process_offline(np.concatenate(offline_samples) if offline_samples else np.empty(0))
            else:
                for rendered in processor.finish():
                    if rendered:
                        yield rendered
        metrics = processor.metrics
        app.state.metrics.values["aegis_audio_received_seconds_total"] += metrics.audio_received_samples / SAMPLE_RATE
        app.state.metrics.values["aegis_audio_processed_seconds_total"] += metrics.audio_processed_samples / SAMPLE_RATE
        app.state.metrics.values["aegis_audio_emitted_seconds_total"] += metrics.audio_emitted_samples / SAMPLE_RATE
        app.state.metrics.values["aegis_segments_natural_total"] += metrics.natural_segments
        app.state.metrics.values["aegis_segments_hard_seam_total"] += metrics.hard_seams
    except PcmProtocolError:
        app.state.metrics.values["aegis_pcm_invalid_residual_bytes_total"] += 1
        raise
    except asyncio.CancelledError:
        app.state.metrics.values["aegis_synthesis_cancellations_total"] += 1
        raise
    finally:
        await upstream.aclose()
        _release_slot()


@app.get("/", include_in_schema=False)
async def web_player() -> HTMLResponse:
    """Same-origin browser player for raw PCM streaming; it exposes no API secrets."""
    return HTMLResponse(WEB_PLAYER, headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health() -> Response:
    # Liveness deliberately does not depend on the model or protected assets.
    return Response(status_code=200, headers=_runtime_headers())


@app.get("/health/ready")
@app.get("/ready")
async def ready() -> Response:
    if not await _backend_healthy():
        return Response(status_code=503, headers=_runtime_headers())
    return Response(status_code=200 if await _warmup() else 503, headers=_runtime_headers())


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    service = app.state.service
    active = 1 if service.active_id else 0
    queued = sum(1 for item in service.runtimes.values() if item.request.status == RequestStatus.QUEUED)
    buffered = sum(item.buffered_bytes for item in service.runtimes.values()) / (SAMPLE_RATE * 2)
    app.state.metrics.values.update({
        "aegis_tts_requests_total": service.metrics["requests_total"],
        "aegis_tts_requests_active": active,
        "aegis_tts_requests_queued": queued,
        "aegis_tts_requests_completed_total": service.metrics["completed_total"],
        "aegis_tts_requests_cancelled_total": service.metrics["cancelled_total"],
        "aegis_tts_requests_failed_total": service.metrics["failed_total"],
        "aegis_tts_segments_queued": len(service._heap),
        "aegis_tts_segments_generated_total": service.metrics["segments_generated_total"],
        "aegis_tts_pcm_buffered_seconds": buffered,
        "aegis_tts_client_stall_total": service.metrics["client_stall_total"],
        "aegis_tts_queue_wait_seconds": service.metrics["queue_wait_seconds"],
    })
    return PlainTextResponse(app.state.metrics.render())


@app.get("/v1/models")
async def models() -> Response:
    try:
        upstream = await app.state.client.get("/v1/models")
    except httpx.HTTPError:
        return Response(status_code=503)
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))


async def _submit_native(payload: dict[str, Any]) -> Runtime:
    speech = _native_request(payload)
    try:
        normalizing_started = perf_counter()
        result = app.state.normalizer.normalize(speech.original_text)
        app.state.metrics.values["aegis_tts_normalization_seconds"] += perf_counter() - normalizing_started
        speech.normalized_text = result.normalized_text
    except ValueError as exc:
        code = "TEXT_TOO_LONG" if "maximum length" in str(exc) else "NORMALIZATION_FAILED"
        raise _native_error(422, code, speech.request_id) from exc
    try:
        segmentation_started = perf_counter()
        drafts = app.state.segmenter.segment(result)
        app.state.metrics.values["aegis_tts_segmentation_seconds"] += perf_counter() - segmentation_started
    except ValueError as exc:
        raise _native_error(422, "SEGMENTATION_FAILED", speech.request_id) from exc
    segments = [SpeechSegment(speech.request_id, item.segment_index, item.original_text, item.normalized_text, SegmentStatus.QUEUED, max(1.0, len(item.normalized_text) / 80), original_span=item.original_span, normalized_span=item.normalized_span, split_reason=item.split_reason) for item in drafts]
    try:
        return await app.state.service.submit(speech, segments)
    except QueueFullError as exc:
        raise _native_error(429, "QUEUE_FULL", speech.request_id) from exc
    except BusyError as exc:
        raise _native_error(409, "QUEUE_FULL", speech.request_id, "busy") from exc
    except PermissionError as exc:
        raise _native_error(403, "INVALID_REQUEST", speech.request_id, "replace_all is disabled") from exc
    except ValueError as exc:
        raise _native_error(409, "INVALID_REQUEST", speech.request_id, "duplicate request id") from exc


def _runtime_headers() -> dict[str, str]:
    return {"X-Aegis-TTS-Source-Commit": SOURCE_COMMIT, "X-Aegis-TTS-DSP-Variant": INITIAL_SEGMENT}


def _native_headers(request_id: str, response_format: str) -> dict[str, str]:
    headers = {
        "X-Aegis-Request-Id": request_id,
        "X-Aegis-Voice-Profile": VOICE_PROFILE,
        "X-Aegis-Audio-Format": "pcm_s16le",
        "X-Aegis-Sample-Rate": "24000",
        "X-Aegis-Channels": "1",
        **_runtime_headers(),
    }
    headers["content-type"] = "application/octet-stream" if response_format == "pcm" else "audio/wav"
    return headers


@app.post("/v1/aegis/speech")
async def aegis_speech(payload: dict[str, Any], request: Request) -> Response:
    runtime = await _submit_native(payload)
    headers = _native_headers(runtime.request.request_id, runtime.request.response_format)
    if runtime.request.stream:
        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in app.state.service.audio_stream(runtime):
                    if await request.is_disconnected():
                        await app.state.service.cancel(runtime.request.request_id, "client_disconnected")
                        break
                    yield chunk
            finally:
                if not runtime.done.is_set():
                    await app.state.service.cancel(runtime.request.request_id, "client_disconnected")
        return StreamingResponse(body(), headers=headers)
    pcm = b"".join([chunk async for chunk in app.state.service.audio_stream(runtime)])
    if runtime.request.status == RequestStatus.FAILED:
        raise _native_error(502, runtime.request.error_code or "UPSTREAM_GENERATION_FAILED", runtime.request.request_id)
    return Response(content=pcm if runtime.request.response_format == "pcm" else _pcm_wav(pcm), headers=headers)


@app.get("/v1/aegis/speech/{request_id}")
async def aegis_status(request_id: str) -> dict[str, Any]:
    runtime = app.state.service.runtimes.get(request_id)
    if not runtime:
        raise _native_error(404, "REQUEST_NOT_FOUND", request_id)
    item = runtime.request
    return {"request_id": item.request_id, "status": item.status.value, "priority": item.priority, "stream": item.stream, "response_format": item.response_format, "error_code": item.error_code, "segments": len(runtime.segments)}


@app.delete("/v1/aegis/speech/{request_id}")
async def aegis_cancel(request_id: str) -> dict[str, Any]:
    runtime = await app.state.service.cancel(request_id)
    if not runtime:
        raise _native_error(404, "REQUEST_NOT_FOUND", request_id)
    return {"request_id": request_id, "status": runtime.request.status.value}


@app.get("/v1/aegis/speech/{request_id}/events")
async def aegis_events(request_id: str, request: Request) -> StreamingResponse:
    runtime = app.state.service.runtimes.get(request_id)
    if not runtime:
        raise _native_error(404, "REQUEST_NOT_FOUND", request_id)
    try: after = int(request.headers.get("Last-Event-ID", "0"))
    except ValueError: after = 0
    async def body() -> AsyncIterator[bytes]:
        emitted = after
        while True:
            for event in list(runtime.events):
                if event.sequence > emitted:
                    emitted = event.sequence
                    import json
                    payload = {"event": event.event, "request_id": event.request_id, "timestamp": event.timestamp, "sequence": event.sequence, **event.data}
                    yield f"id: {event.sequence}\nevent: {event.event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
            if runtime.done.is_set(): break
            await asyncio.sleep(0.2)
    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


@app.get("/v1/aegis/queue")
async def aegis_queue() -> dict[str, Any]:
    service = app.state.service
    return {"active_request_id": service.active_id, "queued_requests": sum(1 for item in service.runtimes.values() if item.request.status == RequestStatus.QUEUED), "queued_segments": len(service._heap), "limits": {"max_requests": service.max_requests, "max_segments": service.max_segments, "max_total_characters": service.max_characters}}


@app.post("/v1/audio/speech")
async def speech(payload: dict[str, Any], request: Request) -> Response:
    """OpenAI compatibility uses the same bounded scheduler as the native API."""
    text = payload.get("input", payload.get("text", ""))
    stream = bool(payload.get("stream", False))
    response_format = str(payload.get("response_format", "wav")).lower()
    if response_format not in {"wav", "pcm"}:
        raise HTTPException(status_code=422, detail={"code": "UNSUPPORTED_FORMAT"})
    runtime = await _submit_native({"text": text, "priority": 50, "interrupt_policy": "enqueue", "stream": stream, "response_format": response_format, "voice_profile": NATIVE_PROFILE})
    headers = _native_headers(runtime.request.request_id, response_format)
    headers["X-Aegis-Voice-Profile"] = VOICE_PROFILE
    if stream:
        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in app.state.service.audio_stream(runtime):
                    if await request.is_disconnected():
                        await app.state.service.cancel(runtime.request.request_id, "client_disconnected")
                        break
                    yield chunk
            finally:
                if not runtime.done.is_set():
                    await app.state.service.cancel(runtime.request.request_id, "client_disconnected")
        return StreamingResponse(body(), headers=headers)
    pcm = b"".join([chunk async for chunk in app.state.service.audio_stream(runtime)])
    if runtime.request.status == RequestStatus.FAILED:
        raise HTTPException(status_code=502, detail={"code": runtime.request.error_code or "UPSTREAM_GENERATION_FAILED", "request_id": runtime.request.request_id})
    return Response(content=pcm if response_format == "pcm" else _pcm_wav(pcm), headers=headers)
