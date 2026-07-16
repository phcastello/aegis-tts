"""One-process GPU model lifetime, prompt cache, and serialized synthesis.

The Qwen model is intentionally loaded only during startup.  This module never
uses a model ID as a fallback: absence of the explicit local checkpoint leaves
the HTTP process alive in a degraded state instead of downloading anything.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import soundfile as sf

from .config import AppSettings
from .metrics import GpuMemory, MetricsCollector, real_time_factor
from .reference_repository import (
    ReferenceNotFoundError,
    ReferenceRepository,
    ReferenceValidationError,
    VoiceReference,
)
from .schemas import SynthesisRequest


LOGGER = logging.getLogger("aegis_tts.model")


class ServiceError(RuntimeError):
    """Base error whose message is safe for the API after sanitization."""


class ServiceNotReadyError(ServiceError):
    pass


class UnknownVoiceError(ServiceError):
    pass


class VoiceUnavailableError(ServiceError):
    pass


class QueueTimeoutError(ServiceError):
    pass


class GenerationTimeoutError(ServiceError):
    pass


class GpuOutOfMemoryError(ServiceError):
    pass


class ModelMissingError(ServiceError):
    pass


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """A complete WAV response generated without retaining its text in memory."""

    wav_bytes: bytes
    headers: dict[str, str]


@dataclass(frozen=True, slots=True)
class _GenerationStats:
    voice_id: str
    seed: int | None
    sample_rate: int
    audio_seconds: float
    generation_seconds: float
    queue_wait_seconds: float
    memory_before: GpuMemory
    memory_after: GpuMemory


ModelFactory = Callable[..., Any]


class ModelService:
    """Own one Base model, reusable ICL clone prompts, and one GPU queue."""

    def __init__(
        self,
        settings: AppSettings,
        repository: ReferenceRepository,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self._model_factory = model_factory or _load_qwen_model
        self._metrics = MetricsCollector(settings.device)
        self._model: Any | None = None
        self.clone_prompts: dict[str, Any] = {}
        self._references: dict[str, VoiceReference] = {}
        self._voice_states: dict[str, str] = {}
        self._voice_errors: dict[str, str] = {}
        self._state = "starting"
        self._errors: list[str] = []
        self._effective_dtype = settings.dtype
        self._load_seconds: float | None = None
        self._prompt_seconds: dict[str, float] = {}
        self._started = False
        self._stopped = False
        self._startup_task: asyncio.Task[None] | None = None
        self._startup_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._sync_inference_lock = threading.Lock()
        # Keeping model load and all CUDA calls on one worker is less surprising
        # for CUDA contexts than the generic asyncio thread pool.  The explicit
        # threading.Lock remains as a second layer of protection.
        self._cuda_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aegis-tts-cuda")
        self._worker_closed = False
        self._queue_waiters = 0

    @property
    def default_voice_id(self) -> str | None:
        try:
            return self.repository.default_voice_id
        except ReferenceValidationError:
            return None

    @property
    def is_ready(self) -> bool:
        default_voice = self.default_voice_id
        return bool(
            self._metrics.cuda_available()
            and self._model is not None
            and default_voice
            and default_voice in self.clone_prompts
            and self._state == "ready"
        )

    def start_background(self) -> asyncio.Task[None]:
        """Schedule non-blocking startup so live/health endpoints stay useful."""

        if self._startup_task is None or self._startup_task.done():
            self._startup_task = asyncio.create_task(self.start(), name="aegis-tts-startup")
        return self._startup_task

    async def start(self) -> None:
        """Load the local Base checkpoint and build each valid prompt once."""

        async with self._startup_lock:
            if self._started or self._stopped:
                return
            self._started = True
            started = time.perf_counter()
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._cuda_worker, self._startup_sync)
            except Exception as exc:  # _startup_sync normally converts errors
                self._state = "error"
                self._add_error("falha inesperada ao iniciar o modelo")
                LOGGER.exception("falha inesperada na inicialização do modelo")
            finally:
                self._load_seconds = round(time.perf_counter() - started, 4)

    async def stop(self) -> None:
        """Wait for an in-flight CUDA call before releasing process resources."""

        if self._stopped:
            return
        self._stopped = True
        # A generation that exceeded the HTTP timeout still owns the worker and
        # must finish before process teardown; do not cancel CUDA unsafely.
        if not self._worker_closed:
            self._worker_closed = True
            await asyncio.to_thread(self._cuda_worker.shutdown, True, cancel_futures=False)
        self.clone_prompts.clear()
        self._references.clear()
        self._model = None
        self._metrics.empty_cache()

    def health_snapshot(self) -> dict[str, Any]:
        gpu = self._metrics.gpu_info()
        return {
            "status": self._state,
            "model": self.settings.model_id,
            "model_loaded": self._model is not None,
            "device": self.settings.device,
            "dtype": self._effective_dtype,
            "attention": self.settings.attention_implementation,
            "cache_implementation": self.settings.cache_implementation,
            "voices": self._health_voice_states(),
            "gpu": gpu,
            "load_seconds": self._load_seconds,
            "prompt_seconds": dict(self._prompt_seconds),
            "errors": list(self._errors),
        }

    def info(self) -> dict[str, Any]:
        return {
            "service": "aegis-tts",
            "model": self.settings.model_id,
            "device": self.settings.device,
            "dtype": self._effective_dtype,
            "attention": self.settings.attention_implementation,
            "cache_implementation": self.settings.cache_implementation,
            "max_text_length": self.settings.max_text_length,
            "default_voice": self.default_voice_id,
        }

    def voices_response(self) -> dict[str, Any]:
        try:
            default_voice = self.repository.default_voice_id
            configs = self.repository.voice_configs(enabled_only=True)
        except ReferenceValidationError as exc:
            raise ServiceNotReadyError("catálogo de vozes indisponível") from exc

        voices: list[dict[str, Any]] = []
        for voice in configs:
            reference = self._references.get(voice.id)
            metadata = reference.metadata if reference else None
            voices.append(
                {
                    "id": voice.id,
                    "display_name": voice.display_name,
                    "ready": voice.id in self.clone_prompts,
                    "reference_duration_seconds": metadata.duration_seconds if metadata else None,
                    "reference_sample_rate": metadata.sample_rate if metadata else None,
                }
            )
        return {"default_voice": default_voice, "voices": voices}

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Queue one request and retain GPU ownership through real completion."""

        voice_id = request.voice or self.default_voice_id
        if not voice_id:
            raise ServiceNotReadyError("voz padrão indisponível")
        self._assert_voice_available(voice_id)
        if self._model is None:
            raise ServiceNotReadyError("modelo ainda não foi carregado")

        queued_at = time.perf_counter()
        self._queue_waiters += 1
        acquired = False
        try:
            try:
                await asyncio.wait_for(
                    self._inference_lock.acquire(),
                    timeout=self.settings.queue_timeout_seconds,
                )
                acquired = True
            except TimeoutError as exc:
                raise QueueTimeoutError("fila de síntese excedeu o tempo limite") from exc
        finally:
            self._queue_waiters = max(0, self._queue_waiters - 1)

        queue_wait = time.perf_counter() - queued_at
        worker_future: asyncio.Future[tuple[bytes, _GenerationStats]] | None = None
        release_on_completion = False
        try:
            loop = asyncio.get_running_loop()
            worker_future = asyncio.wrap_future(
                self._cuda_worker.submit(self._generate_sync, request, voice_id, queue_wait),
                loop=loop,
            )
            try:
                wav_bytes, stats = await asyncio.wait_for(
                    asyncio.shield(worker_future),
                    timeout=self.settings.generation_timeout_seconds,
                )
            except TimeoutError as exc:
                release_on_completion = True
                raise GenerationTimeoutError(
                    "a geração excedeu o tempo limite; a GPU continua ocupada até concluir"
                ) from exc
            except asyncio.CancelledError:
                # Client disconnects/cancellation cannot make a synchronous CUDA
                # call safe to interrupt.  Keep the async lock until its future
                # actually resolves, then let the next queued request continue.
                release_on_completion = True
                raise
            return SynthesisResult(
                wav_bytes=wav_bytes,
                headers=_headers_from_stats(
                    stats, self._effective_dtype, self.settings.model_id
                ),
            )
        finally:
            if acquired:
                if release_on_completion and worker_future is not None and not worker_future.done():
                    worker_future.add_done_callback(self._release_after_background_generation)
                else:
                    self._release_inference_lock()

    def _startup_sync(self) -> None:
        if not self._metrics.cuda_available():
            self._state = "error"
            self._add_error("CUDA não está disponível")
            return
        try:
            self._state = "loading_model"
            _assert_local_model_present(Path(self.settings.model_dir), self.settings.model_id)
            torch, dtype = _select_torch_dtype(self.settings.dtype)
            self._effective_dtype = "bfloat16" if dtype is getattr(torch, "bfloat16", None) else "float16"
            self._model = self._model_factory(
                str(self.settings.model_dir),
                device_map=self.settings.device,
                dtype=dtype,
                attn_implementation=self.settings.attention_implementation,
                local_files_only=True,
            )
        except ModelMissingError as exc:
            self._state = "error"
            self._add_error(str(exc))
            return
        except Exception:
            self._state = "error"
            self._add_error("não foi possível carregar o modelo local")
            LOGGER.exception("falha no carregamento do modelo local")
            return

        self._state = "loading_references"
        try:
            # Validate all enabled profiles first.  Invalid optional comparison
            # profiles do not prevent an otherwise valid default from serving.
            references = self.repository.list_references()
            for config in self.repository.voice_configs(enabled_only=True):
                reference = references.get(config.id)
                if reference is None:
                    message = self.repository.reference_error(config.id) or "referência inválida"
                    self._voice_states[config.id] = "error"
                    self._voice_errors[config.id] = message
                    self._add_error(f"voz {config.id}: {message}")
                    continue
                self._references[config.id] = reference
                self._build_prompt_sync(config.id, reference)
        except ReferenceValidationError as exc:
            self._state = "degraded"
            self._add_error("catálogo de vozes inválido")
            LOGGER.warning("catálogo de vozes indisponível: %s", type(exc).__name__)
            return

        if self.default_voice_id in self.clone_prompts:
            self._state = "ready"
        else:
            self._state = "degraded"
            self._add_error("a voz padrão não ficou pronta")

    def _build_prompt_sync(self, voice_id: str, reference: VoiceReference) -> None:
        started = time.perf_counter()
        try:
            prompt = self._model.create_voice_clone_prompt(
                ref_audio=str(reference.audio_path),
                ref_text=reference.transcript,
                x_vector_only_mode=False,
            )
            self.clone_prompts[voice_id] = prompt
            self._voice_states[voice_id] = "ready"
            elapsed = round(time.perf_counter() - started, 4)
            self._prompt_seconds[voice_id] = elapsed
            # Deliberately log no path, waveform, or transcript.  The hash is
            # sufficient to correlate an operator's reference validation report.
            LOGGER.info(
                "prompt de clone pronto",
                extra={
                    "voice": voice_id,
                    "seconds": elapsed,
                    "duration_seconds": reference.metadata.duration_seconds,
                    "audio_sha256": reference.metadata.audio_sha256,
                },
            )
        except Exception:
            self._voice_states[voice_id] = "error"
            self._voice_errors[voice_id] = "falha ao criar prompt de clone"
            self._add_error(f"voz {voice_id}: falha ao criar prompt de clone")
            # Avoid the traceback here: some third-party error messages include
            # a reference pathname.  The voice ID identifies the actionable item.
            LOGGER.error(
                "falha ao criar prompt de clone para a voz %s",
                voice_id,
                extra={
                    "voice": voice_id,
                    "seconds": round(time.perf_counter() - started, 4),
                    "duration_seconds": reference.metadata.duration_seconds,
                    "audio_sha256": reference.metadata.audio_sha256,
                    "success": False,
                },
            )

    def _assert_voice_available(self, voice_id: str) -> None:
        try:
            configured_ids = {voice.id for voice in self.repository.voice_configs(enabled_only=True)}
        except ReferenceValidationError as exc:
            raise ServiceNotReadyError("catálogo de vozes indisponível") from exc
        if voice_id not in configured_ids:
            raise UnknownVoiceError("voice id desconhecido")
        if voice_id not in self.clone_prompts:
            raise VoiceUnavailableError("a voz solicitada ainda não está pronta")

    def _generate_sync(
        self,
        request: SynthesisRequest,
        voice_id: str,
        queue_wait: float,
    ) -> tuple[bytes, _GenerationStats]:
        """Synchronous CUDA section; called only from the dedicated worker."""

        with self._sync_inference_lock:
            if self._model is None or voice_id not in self.clone_prompts:
                raise ServiceNotReadyError("modelo ou prompt não está pronto")
            _seed_everything(request.seed)
            self._metrics.reset_peak()
            before = self._metrics.memory()
            started = time.perf_counter()
            try:
                params = request.generation_parameters.model_dump(exclude_none=True)
                generate_kwargs = {
                    "text": request.text,
                    "language": request.language,
                    "voice_clone_prompt": self.clone_prompts[voice_id],
                    # This HTTP endpoint returns a complete WAV, not streamed
                    # chunks. Avoid Qwen's text-input streaming simulation.
                    "non_streaming_mode": True,
                    **params,
                }
                if self.settings.cache_implementation:
                    # Qwen forwards these kwargs to Transformers generate().
                    # Keep this experiment opt-in: StaticCache preallocates
                    # VRAM and may trigger compilation on compatible models.
                    generate_kwargs.update(
                        use_cache=True,
                        cache_implementation=self.settings.cache_implementation,
                    )
                if self.settings.profile_generation:
                    wavs, sample_rate = self._profile_generation(generate_kwargs)
                else:
                    wavs, sample_rate = self._model.generate_voice_clone(**generate_kwargs)
                waveform = _one_mono_waveform(wavs)
                sample_rate = int(sample_rate)
                if sample_rate <= 0:
                    raise ServiceError("o modelo retornou sample rate inválido")
                wav_bytes = _encode_pcm16(waveform, sample_rate)
            except Exception as exc:
                if _is_cuda_oom(exc):
                    memory_at_oom = self._metrics.memory()
                    self._metrics.empty_cache()
                    memory_after_cleanup = self._metrics.memory()
                    LOGGER.warning(
                        "OOM CUDA durante síntese",
                        extra={
                            "voice": voice_id,
                            "vram_before_mib": memory_at_oom.allocated_mib,
                            "vram_reserved_before_mib": memory_at_oom.reserved_mib,
                            "vram_peak_before_mib": memory_at_oom.peak_allocated_mib,
                            "vram_after_mib": memory_after_cleanup.allocated_mib,
                            "vram_reserved_after_mib": memory_after_cleanup.reserved_mib,
                        },
                    )
                    raise GpuOutOfMemoryError("VRAM insuficiente para concluir a síntese") from exc
                if isinstance(exc, ServiceError):
                    raise
                # Do not include request.text in an error or log record.
                LOGGER.exception("falha de geração para a voz %s", voice_id)
                raise ServiceError("falha ao gerar o áudio") from exc
            generation_seconds = time.perf_counter() - started
            after = self._metrics.memory()
            audio_seconds = float(waveform.shape[0]) / sample_rate
            stats = _GenerationStats(
                voice_id=voice_id,
                seed=request.seed,
                sample_rate=sample_rate,
                audio_seconds=audio_seconds,
                generation_seconds=generation_seconds,
                queue_wait_seconds=queue_wait,
                memory_before=before,
                memory_after=after,
            )
            LOGGER.info(
                "síntese concluída",
                extra={
                    "voice": voice_id,
                    "seed": request.seed,
                    "generation_seconds": round(generation_seconds, 4),
                    "audio_seconds": round(audio_seconds, 4),
                    "rtf": real_time_factor(generation_seconds, audio_seconds),
                    "queue_wait_seconds": round(queue_wait, 4),
                },
            )
            if self.settings.save_samples:
                self._save_sample(wav_bytes, request, stats)
            return wav_bytes, stats

    def _profile_generation(self, generate_kwargs: dict[str, Any]) -> tuple[Any, Any]:
        """Profile one explicitly opted-in generation and save its Chrome trace.

        This is intentionally controlled solely by an environment setting that
        defaults to false.  Operators should enable it for one request only,
        because the profiler materially changes generation timing.
        """

        import torch
        from torch.profiler import ProfilerActivity, profile

        trace_path = Path(self.settings.samples_dir) / "qwen-profile.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        torch.cuda.synchronize()
        with profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            result = self._model.generate_voice_clone(**generate_kwargs)
        torch.cuda.synchronize()
        cuda_table = profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=40
        )
        cpu_table = profiler.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=40
        )
        profiler.export_chrome_trace(str(trace_path))
        LOGGER.info("PyTorch profiler CUDA (top 40):\n%s", cuda_table)
        LOGGER.info("PyTorch profiler CPU (top 40):\n%s", cpu_table)
        LOGGER.info("trace do PyTorch profiler salva em %s", trace_path)
        return result

    def _save_sample(self, wav_bytes: bytes, request: SynthesisRequest, stats: _GenerationStats) -> None:
        """Persist an opt-in sample and bounded non-sensitive metadata only."""

        try:
            samples_dir = Path(self.settings.samples_dir)
            samples_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            stem = f"{stamp}-{stats.voice_id}-{uuid4().hex}"
            wav_path = samples_dir / f"{stem}.wav"
            json_path = samples_dir / f"{stem}.json"
            temp_wav = samples_dir / f".{stem}.wav.tmp"
            temp_json = samples_dir / f".{stem}.json.tmp"
            temp_wav.write_bytes(wav_bytes)
            metadata = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "voice": stats.voice_id,
                "seed": stats.seed,
                "text_length": len(request.text),
                "text_sha256": hashlib.sha256(request.text.encode("utf-8")).hexdigest(),
                "language": request.language,
                "generation_parameters": request.generation_parameters.model_dump(),
                "sample_rate": stats.sample_rate,
                "audio_duration_seconds": round(stats.audio_seconds, 4),
                "generation_seconds": round(stats.generation_seconds, 4),
                "rtf": real_time_factor(stats.generation_seconds, stats.audio_seconds),
            }
            temp_json.write_text(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            temp_wav.replace(wav_path)
            temp_json.replace(json_path)
            retain_samples(samples_dir, self.settings.max_samples)
        except Exception:
            # Saved samples are optional; an unwritable volume must not turn a
            # successful synth into an API failure.  No text/path is logged.
            LOGGER.warning("não foi possível salvar o sample opcional")

    def _release_after_background_generation(self, future: asyncio.Future[Any]) -> None:
        try:
            # Retrieve an unobserved worker exception after HTTP timeout so it
            # does not become an asyncio warning.  Detailed failure stays in logs.
            future.exception()
        except (asyncio.CancelledError, Exception):
            pass
        self._release_inference_lock()

    def _release_inference_lock(self) -> None:
        if self._inference_lock.locked():
            self._inference_lock.release()

    def _health_voice_states(self) -> dict[str, str]:
        try:
            states = self.repository.status_by_voice()
        except ReferenceValidationError:
            return dict(self._voice_states)
        for voice_id, state in self._voice_states.items():
            states[voice_id] = state
        return states

    def _add_error(self, message: str) -> None:
        if message not in self._errors:
            self._errors.append(message[:300])


def _load_qwen_model(model_path: str, **kwargs: Any) -> Any:
    """Late import avoids importing Qwen/Torch in ordinary unit tests."""

    # Defense in depth: even a direct ``python -m uvicorn`` invocation cannot
    # turn a missing local checkpoint into an implicit Hub fetch.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(model_path, **kwargs)


def _download_script_for(model_id: str) -> str:
    """Return the explicit downloader corresponding to an approved model."""

    return "scripts/download-model-0.6b.sh" if model_id.endswith("0.6B-Base") else "scripts/download-model.sh"


def _has_primary_weights(model_dir: Path) -> bool:
    """Accept an unsharded safetensors checkpoint or a complete shard index."""

    if (model_dir / "model.safetensors").is_file():
        return True
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        filenames = set(index.get("weight_map", {}).values())
    except (OSError, ValueError, AttributeError):
        return False
    return bool(filenames) and all((model_dir / filename).is_file() for filename in filenames)


def _assert_local_model_present(model_dir: Path, model_id: str) -> None:
    downloader = _download_script_for(model_id)
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise ModelMissingError(f"modelo local ausente; execute {downloader}")
    if not _has_primary_weights(model_dir):
        raise ModelMissingError(f"pesos principais do modelo ausentes; execute {downloader}")


def _select_torch_dtype(requested: str) -> tuple[Any, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - image build catches this
        raise ServiceError("PyTorch CUDA não está disponível") from exc
    if requested == "bfloat16" and bool(torch.cuda.is_bf16_supported()):
        return torch, torch.bfloat16
    return torch, torch.float16


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    # Qwen uses torch sampling; setting NumPy too avoids hidden nondeterminism
    # in surrounding audio utilities.  The request queue serializes this global
    # state so independent requests cannot race it.
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


def _one_mono_waveform(wavs: Any) -> np.ndarray:
    if not isinstance(wavs, (list, tuple)) or not wavs:
        raise ServiceError("o modelo não retornou áudio")
    waveform = np.asarray(wavs[0], dtype=np.float32)
    if waveform.ndim == 2 and 1 in waveform.shape:
        waveform = waveform.reshape(-1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ServiceError("o modelo retornou áudio não-mono inválido")
    if not np.isfinite(waveform).all():
        raise ServiceError("o modelo retornou amostras de áudio inválidas")
    return waveform


def _encode_pcm16(waveform: np.ndarray, sample_rate: int) -> bytes:
    output = BytesIO()
    # This is a container conversion required by the HTTP contract (PCM 16-bit),
    # not normalization, resampling, EQ, compression, or an audio effect.
    sf.write(output, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _is_cuda_oom(error: Exception) -> bool:
    try:
        import torch

        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(error, oom_type):
            return True
    except Exception:
        pass
    return False


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _headers_from_stats(stats: _GenerationStats, dtype: str, model_id: str) -> dict[str, str]:
    rtf = real_time_factor(stats.generation_seconds, stats.audio_seconds)
    return {
        "X-Aegis-TTS-Model": model_id,
        "X-Aegis-TTS-Voice": stats.voice_id,
        "X-Aegis-TTS-Seed": str(stats.seed) if stats.seed is not None else "random",
        "X-Aegis-TTS-Generation-Seconds": _number(stats.generation_seconds),
        "X-Aegis-TTS-Audio-Duration-Seconds": _number(stats.audio_seconds),
        "X-Aegis-TTS-RTF": _number(rtf),
        "X-Aegis-TTS-Queue-Wait-Seconds": _number(stats.queue_wait_seconds),
        "X-Aegis-TTS-Sample-Rate": str(stats.sample_rate),
        "X-Aegis-TTS-Dtype": dtype,
        "X-Aegis-TTS-VRAM-Before-MiB": _number(stats.memory_before.allocated_mib),
        "X-Aegis-TTS-VRAM-After-MiB": _number(stats.memory_after.allocated_mib),
        "X-Aegis-TTS-VRAM-Peak-MiB": _number(stats.memory_after.peak_allocated_mib),
    }


def retain_samples(samples_dir: Path, max_samples: int) -> None:
    """Delete only oldest generated root-level sample pairs, never references.

    Smoke-test output lives in ``samples/smoke`` and is deliberately excluded.
    Files not ending in the service's generated WAV/JSON pairing are untouched.
    """

    if max_samples < 0 or not samples_dir.is_dir():
        return
    wavs = sorted(
        (path for path in samples_dir.glob("*.wav") if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    for wav_path in wavs[:-max_samples] if max_samples else wavs:
        # Only remove the JSON name paired with a WAV selected for retention.
        # Do not recurse and never follow/remove anything outside samples_dir.
        json_path = wav_path.with_suffix(".json")
        try:
            wav_path.unlink(missing_ok=True)
            json_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("não foi possível remover um sample antigo")
