#!/usr/bin/env python3
"""Isolated, local-only VoxCPM2 streaming runtime; default action is a plan.

This does not download weights, install dependencies, train, stop production or
change endpoints. Execute only within an owner-approved exclusive GPU window.
The source checkout and dependency environment must already exist separately.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    from .benchmark import (gpu_snapshot, json_bytes, load_json, private_directory,
                            validate_endpoint, validate_window, write_private)
except ImportError:
    from benchmark import (gpu_snapshot, json_bytes, load_json, private_directory,
                           validate_endpoint, validate_window, write_private)

SOURCE_COMMIT = "cce58a5b59303c9bd63d12f23afe6a49a4a80c59"
WEIGHTS = {
    "model.safetensors": "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d",
    "audiovae.pth": "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1",
}
METADATA_GIT_BLOBS = {
    "config.json": "792f1c223ed607f9e508c0a0deb15dd9532483be",
    "special_tokens_map.json": "8619dda6f3eb6d60d0a1bb274820054e46f41699",
    "tokenization_voxcpm2.py": "e7d768677298d058fa6ef8b160e3ca4430997fad",
    "tokenizer.json": "41a5c2a8dba4058dd1ad73fb898abf5e4f64f0f9",
    "tokenizer_config.json": "fecf4cdae73b57053cac2ad34c67febbe4e4f08b",
}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require_empty_gpu(snapshot):
    if "A2000" not in snapshot["name"] or not 11000 <= snapshot["total_mib"] <= 12288:
        raise ValueError("expected the RTX A2000 12GB")
    if snapshot["processes"] or snapshot["used_mib"] > 512:
        raise ValueError("GPU is occupied; arrange the approved window manually")


def require_own_gpu(snapshot):
    actual = {item["pid"] for item in snapshot["processes"]}
    if actual != {os.getpid()}:
        raise ValueError("GPU is no longer exclusive to this runtime")


def validate_local_files(args):
    source = Path(args.source_dir).expanduser().resolve(strict=True)
    model = Path(args.model_dir).expanduser().resolve(strict=True)
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            text=True, capture_output=True, timeout=5, check=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
                            text=True, capture_output=True, timeout=5, check=True).stdout.strip()
    if commit != SOURCE_COMMIT or status:
        raise ValueError("VoxCPM source checkout must be clean and match the pinned source commit")
    # Upstream prioritizes this alternate decoder if present. The pinned
    # artifact is the PTH below, so do not silently validate one and load another.
    for alternate in ("audiovae.safetensors", "pytorch_model.bin"):
        if (model / alternate).exists():
            raise ValueError("remove alternate unpinned " + alternate + " from the isolated model directory")
    for name, expected in WEIGHTS.items():
        if file_hash(model / name) != expected:
            raise ValueError("VoxCPM2 weight SHA256 mismatch: " + name)
    for name, expected in METADATA_GIT_BLOBS.items():
        contents = (model / name).read_bytes()
        header = f"blob {len(contents)}\0".encode("ascii")
        if hashlib.sha1(header + contents).hexdigest() != expected:
            raise ValueError("VoxCPM2 metadata Git blob mismatch: " + name)
    config = load_json(model / "config.json")
    vae = config["audio_vae_config"]
    if (config.get("architecture"), config.get("dtype"), vae.get("sample_rate"), vae.get("out_sample_rate")) != (
            "voxcpm2", "bfloat16", 16000, 48000):
        raise ValueError("unexpected architecture, dtype or AudioVAE sample rates")
    reference = Path(args.reference_wav).expanduser().resolve(strict=True)
    transcript = Path(args.reference_text_file).expanduser().resolve(strict=True)
    text = transcript.read_text(encoding="utf-8").strip()
    if not text or len(text) > 8192:
        raise ValueError("reference transcript is empty or exceeds 8192 characters")
    return source, model, reference, text, {
        "source_commit": commit, "weights": WEIGHTS, "metadata_git_blobs": METADATA_GIT_BLOBS,
        "config_sha256": file_hash(model / "config.json"),
        "reference_sha256": file_hash(reference),
        "reference_transcript_sha256": file_hash(transcript),
    }


def pcm16_bytes(values, np):
    """Explicit float-to-PCM conversion; refuse out-of-range or nonfinite output."""
    values = np.asarray(values)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("model returned nonfinite or non-mono audio")
    if values.size and (values.min() < -1.0 or values.max() > 1.0):
        raise ValueError("model output exceeds [-1,1]; refusing silent clipping")
    return np.rint(values * 32767.0).astype("<i2").tobytes()


class Runtime:
    def __init__(self, args, window, output, model, reference, text, torch, np):
        self.args, self.window, self.output = args, window, output
        self.model, self.reference, self.reference_text = model, reference, text
        self.torch, self.np = torch, np
        self.fatal = False

    def guard(self, check_gpu=True):
        validate_window(load_json(self.args.window_file), self.args.candidate)
        if check_gpu:
            require_own_gpu(gpu_snapshot())

    def synthesize(self, payload, handler):
        guard_started = time.perf_counter()
        self.guard()
        guard_seconds = time.perf_counter() - guard_started
        if payload.get("stream") is not True or payload.get("response_format") != "pcm":
            raise ValueError("runtime requires stream=true and response_format=pcm")
        text = payload.get("input")
        if not isinstance(text, str) or not text.strip() or len(text) > self.args.max_text_chars:
            raise ValueError("input must be a nonempty bounded string")
        if set(payload) - {"input", "stream", "response_format", "model", "seed"}:
            raise ValueError("unsupported payload keys; runtime settings are pinned by CLI")
        seed = payload.get("seed", self.args.seed)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        torch, np = self.torch, self.np
        identifier = uuid.uuid4().hex
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        chunks, first_ms, total_samples = [], None, 0
        result = self.model.generate_streaming(
            text=text, prompt_wav_path=str(self.reference), prompt_text=self.reference_text,
            cfg_value=self.args.cfg_value, inference_timesteps=self.args.timesteps,
            max_len=self.args.max_len, normalize=False, denoise=False,
            retry_badcase=False, seed=seed,
        )
        # Chunked HTTP proves completion: failures omit the terminating zero chunk
        # so clients detect truncation rather than accepting a partial utterance.
        handler.send_response(200)
        handler.send_header("Content-Type", "audio/pcm")
        handler.send_header("X-Aegis-Sample-Rate", "48000")
        handler.send_header("X-Aegis-Channels", "1")
        handler.send_header("X-Aegis-Sample-Format", "s16le")
        handler.send_header("X-Architecture-Request-Id", identifier)
        handler.send_header("Transfer-Encoding", "chunked")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True
        try:
            with torch.inference_mode():
                for values in result:
                    # Native VoxCPM has already copied each VAE output to CPU.
                    self.guard(check_gpu=False)
                    values = np.asarray(values, dtype=np.float32)
                    pcm = pcm16_bytes(values, np)
                    if not pcm:
                        continue
                    if first_ms is None:
                        first_ms = (time.perf_counter() - started) * 1000
                    total_samples += values.size
                    if total_samples > self.args.max_audio_seconds * 48000:
                        raise ValueError("generated audio exceeded configured duration limit")
                    if self.args.save_float:
                        chunks.append(values.copy())
                    handler.wfile.write(f"{len(pcm):x}\r\n".encode("ascii") + pcm + b"\r\n")
                    handler.wfile.flush()
        finally:
            result.close()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if first_ms is None:
            raise ValueError("model returned no PCM")
        metadata = {
            "request_id": identifier, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "effective_seed": seed,
            "conditioning_mode": "continuation",
            "request_guard_seconds": guard_seconds,
            "sample_rate": 48000, "native_dtype": "float32", "wire_format": "s16le-mono-raw",
            "quantization": "round(float32 * 32767), no gain normalization or clipping",
            "first_native_pcm_ms": first_ms, "complete_seconds": elapsed,
            "audio_seconds": total_samples / 48000, "rtf": elapsed / (total_samples / 48000),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "latency_scope": "native generation includes prompt preparation; excludes HTTP client",
            "guard_scope": "HTTP client includes GPU/window checks; native generation timings exclude them",
            "semantic_completion": "not verified by generator API; max_len can end output without EOS; review text and tail",
        }
        # Mark EOF at generation completion. Saving diagnostic float arrays must
        # not inflate client EOF/RTF. Performance runs leave --save-float off;
        # a subsequent request can otherwise wait for the previous artifact IO.
        handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()
        artifact_started = time.perf_counter()
        if self.args.save_float:
            buffer = io.BytesIO()
            np.save(buffer, np.concatenate(chunks), allow_pickle=False)
            write_private(self.output / (identifier + ".float32.npy"), buffer.getvalue())
        metadata["float_artifact_io_seconds"] = time.perf_counter() - artifact_started
        write_private(self.output / (identifier + ".json"), json_bytes(metadata))


def handler_type(runtime):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass  # Never emit request text, reference content or paths.

        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self):
            if self.path != "/v1/audio/speech":
                self.send_error(404)
                return
            self.connection.settimeout(10)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("bounded Content-Length body is required")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("payload must be an object")
                runtime.synthesize(payload, self)
            except Exception as exc:
                # Fail the process after truncation/OOM/cancellation; never expose
                # a truncated stream as a successful measurement. The benchmark
                # records HTTP/stream failures; stderr contains no private values.
                runtime.fatal = True
                self.close_connection = True
                # Keep a private failure receipt even if HTTP failed mid-stream.
                # Avoid serializing the exception message: dependency errors may
                # embed private reference paths or text.
                failure = {"outcome": "failed", "error_type": type(exc).__name__,
                           "phase": "http_request_or_generation", "recorded_unix": time.time()}
                try:
                    failure.update(peak_allocated_mib=runtime.torch.cuda.max_memory_allocated() / 2**20,
                                   peak_reserved_mib=runtime.torch.cuda.max_memory_reserved() / 2**20)
                    write_private(runtime.output / ("failed-" + uuid.uuid4().hex + ".json"),
                                  json_bytes(failure))
                except Exception:
                    pass
                print("VoxCPM request failed: " + type(exc).__name__, file=sys.stderr)
    return Handler


def start_window_watchdog(args):
    """Release this process' CUDA allocations on expiry, including during load.

    This never stops or restarts any other process. The owner still controls the
    production maintenance procedure. The client must retain a restoration margin.
    """
    def monitor():
        while True:
            try:
                remaining = validate_window(load_json(args.window_file), args.candidate)
            except Exception:
                print("VoxCPM operational window ended; exiting runtime", file=sys.stderr, flush=True)
                os._exit(2)
            time.sleep(min(.5, remaining))
    thread = threading.Thread(target=monitor, daemon=True, name="approved-window")
    thread.start()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--candidate", default="voxcpm2-native")
    parser.add_argument("--source-dir")
    parser.add_argument("--model-dir")
    parser.add_argument("--reference-wav")
    parser.add_argument("--reference-text-file")
    parser.add_argument("--window-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--optimize", action="store_true", help="enable torch.compile; measure separately")
    parser.add_argument("--save-float", action="store_true", help="save private native float32 arrays")
    parser.add_argument("--seed", type=int, default=40733133)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--max-audio-seconds", type=int, default=60)
    parser.add_argument("--max-text-chars", type=int, default=1200)
    args = parser.parse_args(argv)
    validate_endpoint(f"http://127.0.0.1:{args.port}/v1/audio/speech")
    if not 1 <= args.timesteps <= 100 or not 0 < args.cfg_value <= 10:
        parser.error("invalid diffusion settings")
    if not 1 <= args.max_len <= 4096 or not 1 <= args.max_audio_seconds <= 300:
        parser.error("invalid generation limits")
    if not 1 <= args.max_text_chars <= 8192:
        parser.error("invalid text limit")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be an unsigned 32-bit integer")
    if args.optimize:
        parser.error("compile experiment disabled: pinned upstream skips optimize for cuda:0; verify actual compiled backend in a separate adapter revision")
    if not args.execute:
        print(json.dumps({"mode": "plan", "candidate": args.candidate,
                          "source_commit": SOURCE_COMMIT, "weights": WEIGHTS,
                          "native_rate": 48000, "reference_encoder_rate": 16000,
                          "streaming": "native autoregressive latents + causal AudioVAE chunks",
                          "optimize": args.optimize, "gpu_touched": False,
                          "required_before_execute": ["owner-approved window JSON", "exclusive A2000",
                                                      "pinned source checkout and installed dependencies",
                                                      "local weights", "private reference and output paths"]}, indent=2))
        return 0
    for name in ("source_dir", "model_dir", "reference_wav", "reference_text_file", "window_file", "output_dir"):
        if not getattr(args, name):
            parser.error("--" + name.replace("_", "-") + " required with --execute")
    os.umask(0o077)
    window = load_json(args.window_file)
    validate_window(window, args.candidate)
    require_empty_gpu(gpu_snapshot())
    output = private_directory(args.output_dir)
    source, weights, reference, text, provenance = validate_local_files(args)
    validate_window(load_json(args.window_file), args.candidate)
    require_empty_gpu(gpu_snapshot())
    start_window_watchdog(args)
    # Imports cannot initialize CUDA until the approved-window and exclusivity
    # checks have passed; all Hub access is disabled, including optional models.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source / "src"))
    import numpy as np
    import torch
    import voxcpm
    if not Path(voxcpm.__file__).resolve().is_relative_to(source / "src"):
        raise ValueError("voxcpm imported from an unexpected source")
    from voxcpm import VoxCPM
    started = time.perf_counter()
    try:
        model = VoxCPM.from_pretrained(str(weights), local_files_only=True, load_denoiser=False,
                                       optimize=args.optimize, device="cuda:0")
        torch.cuda.synchronize()
    except Exception as exc:
        failure = {"outcome": "failed", "phase": "model_load", "error_type": type(exc).__name__,
                   "elapsed_seconds": time.perf_counter() - started,
                   "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                   "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
        write_private(output / ("failed-load-" + uuid.uuid4().hex + ".json"), json_bytes(failure))
        raise
    validate_window(load_json(args.window_file), args.candidate)
    require_own_gpu(gpu_snapshot())
    if model.tts_model.sample_rate != 48000:
        raise ValueError("unexpected native output sample rate")
    provenance.update({"runtime": "VoxCPM2 native PyTorch", "pid": os.getpid(),
                       "loader_seconds": time.perf_counter() - started, "optimize": args.optimize,
                       "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                       "sample_rate": 48000, "response_cache_disabled": True,
                       "loader_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                       "loader_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                       "reference_encoder_rate": 16000,
                       "runtime_script_sha256": file_hash(__file__)})
    write_private(output / ("runtime-" + uuid.uuid4().hex + ".json"), json_bytes(provenance))
    runtime = Runtime(args, window, output, model, reference, text, torch, np)
    server = HTTPServer(("127.0.0.1", args.port), handler_type(runtime))
    server.timeout = 1
    print(json.dumps({"ready": True, "pid": os.getpid(), "port": args.port,
                      "sample_rate": 48000, "candidate": args.candidate}), flush=True)
    try:
        while not runtime.fatal:
            runtime.guard()
            server.handle_request()
    finally:
        server.server_close()
    return 1 if runtime.fatal else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("VoxCPM runtime refused or stopped: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
