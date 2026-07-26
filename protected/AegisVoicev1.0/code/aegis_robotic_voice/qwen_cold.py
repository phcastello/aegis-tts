"""Isolated Qwen 0.6B + strict robotic-resynthesis experiment.

The vLLM 0.22 audio schema exposes only ``seed`` and ``max_new_tokens`` per
request.  Temperature, top-p, top-k and repetition penalty are consequently
written to a private deploy config and each explicit cold profile has one
short-lived, lab-owned vLLM container.  This module never starts, stops or
reconfigures the production service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .analysis.world import analyze_world
from .audio import sha256, validate_audio
from .config import Settings, yaml_config
from .manifests import append_jsonl, latest_candidates, read_jsonl
from .mastering.chain import master
from .synthesis.lpc_engine import analyze_lpc, synthesize_lpc
from .synthesis.world_engine import synthesize_world
from .timing import Timer, stats
from .transforms.envelope import transform_envelope
from .transforms.aperiodicity import transform_aperiodicity
from .transforms.prosody import transform_f0


EXPECTED_MODEL_PATH = "/models/Qwen3-TTS-12Hz-0.6B-Base"
EXPECTED_SAMPLE_RATE = 24000


def _qwen_root(cfg: Settings) -> Path:
    return cfg.root.parents[1]


def _paths(cfg: Settings) -> dict[str, Path]:
    root = cfg.results / "qwen-cold"
    return {
        "root": root,
        "sources": root / "sources.jsonl",
        "candidates": root / "candidates.jsonl",
        "latency": root / "latency.json",
        "reports": cfg.reports / "qwen-cold",
        "runtime": cfg.cache / "qwen-cold-runtime",
    }


def _value_from_environment(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _load_config() -> dict[str, Any]:
    document = yaml_config("qwen-cold.yaml")
    required = {"runtime", "reference", "profiles", "effects"}
    if not required <= document.keys():
        raise ValueError("qwen-cold.yaml is incomplete")
    if set(document["effects"]) != {"L00", "W04"}:
        raise ValueError("the Qwen cold lab must apply exactly L00 and W04")
    for code, profile in document["profiles"].items():
        if not code.startswith("Q") or not {"temperature", "top_p", "top_k", "repetition_penalty"} <= profile.keys():
            raise ValueError(f"invalid cold profile {code}")
    return document


def _reference(config: dict[str, Any]) -> tuple[Path, Path]:
    audio = Path(_value_from_environment("AEGIS_QWEN_COLD_REFERENCE_WAV", config["reference"]["audio"])).resolve()
    transcript = Path(_value_from_environment("AEGIS_QWEN_COLD_REFERENCE_TEXT", config["reference"]["transcript"])).resolve()
    if not audio.is_file() or audio.stat().st_size == 0:
        raise ValueError(f"Aninha reference WAV is unavailable: {audio}")
    if not transcript.is_file() or not transcript.read_text(encoding="utf-8").strip():
        raise ValueError(f"Aninha reference transcript is unavailable or empty: {transcript}")
    return audio, transcript


def _check_prerequisites(cfg: Settings, config: dict[str, Any]) -> None:
    repo = _qwen_root(cfg)
    checkpoint = repo / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
    if not (checkpoint / "config.json").is_file() or not ((checkpoint / "model.safetensors").is_file() or (checkpoint / "model.safetensors.index.json").is_file()):
        raise ValueError(f"Qwen 0.6B safetensors checkpoint is unavailable: {checkpoint}")
    gateway = repo / "gateway" / "run-vllm-with-default-reference.sh"
    if not gateway.is_file():
        raise ValueError(f"vLLM gateway launcher is unavailable: {gateway}")
    _reference(config)
    subprocess.run(["docker", "compose", "version"], check=True, stdout=subprocess.DEVNULL)


def _compose_command(compose: Path, config: dict[str, Any], *args: str) -> list[str]:
    runtime = config["runtime"]
    return ["docker", "compose", "--project-name", runtime["compose_project"], "--file", str(compose), *args]


def _container_running(container: str) -> bool:
    probe = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", container], text=True, capture_output=True)
    return probe.returncode == 0 and probe.stdout.strip() == "true"


def _write_runtime_files(cfg: Settings, config: dict[str, Any], code: str) -> tuple[Path, Path]:
    runtime = config["runtime"]
    profile = config["profiles"][code]
    paths = _paths(cfg)
    work = paths["runtime"] / code
    work.mkdir(parents=True, exist_ok=True)
    deploy = yaml_config("qwen-vllm-template.yaml")
    stage = deploy["stages"][0]["default_sampling_params"]
    for field in ("temperature", "top_p", "top_k", "repetition_penalty"):
        stage[field] = profile[field]
    stage["seed"] = runtime["seed"]
    stage["max_tokens"] = runtime["max_new_tokens"]
    deploy_path = work / "vllm-qwen-cold.yaml"
    deploy_path.write_text(yaml.safe_dump(deploy, sort_keys=False), encoding="utf-8")

    repo = _qwen_root(cfg)
    reference_wav, reference_text = _reference(config)
    port = int(os.environ.get("AEGIS_QWEN_COLD_PORT", runtime["port"]))
    cache = cfg.cache / "qwen-vllm"
    cache.mkdir(parents=True, exist_ok=True)
    compose = {
        "services": {
            runtime["service"]: {
                "image": runtime["image"],
                "container_name": runtime["container_name"],
                "restart": "no",
                "entrypoint": [],
                "command": ["/usr/bin/bash", "/gateway/run-vllm-with-default-reference.sh"],
                "ports": [f"127.0.0.1:{port}:8001"],
                "volumes": [
                    f"{repo / 'models' / 'Qwen3-TTS-12Hz-0.6B-Base'}:{EXPECTED_MODEL_PATH}:ro",
                    f"{reference_wav}:/defaults/reference.wav:ro",
                    f"{reference_text}:/defaults/reference.txt:ro",
                    f"{cache}:/cache",
                    f"{deploy_path}:/config/vllm-qwen3-tts-0.6b.yaml:ro",
                    f"{repo / 'gateway'}:/gateway:ro",
                ],
                "environment": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "NVIDIA_VISIBLE_DEVICES": "0",
                    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                    "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
                    "VLLM_LOGGING_LEVEL": "WARNING",
                    "HF_HOME": "/cache/huggingface",
                    "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub",
                    "TRANSFORMERS_CACHE": "/cache/huggingface/transformers",
                    "XDG_CACHE_HOME": "/cache/xdg",
                    "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
                    "HOME": "/cache/home",
                    "PYTHONPATH": "/gateway",
                    "AEGIS_VLLM_BACKEND_URL": "http://127.0.0.1:8091",
                    "AEGIS_DEFAULT_REFERENCE_AUDIO": "/defaults/reference.wav",
                    "AEGIS_DEFAULT_REFERENCE_TEXT": "/defaults/reference.txt",
                    "AEGIS_DEFAULT_MODEL": EXPECTED_MODEL_PATH,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
                "shm_size": "8gb",
                "security_opt": ["no-new-privileges:true"],
                "deploy": {"resources": {"reservations": {"devices": [{"driver": "nvidia", "device_ids": ["0"], "capabilities": ["gpu"]}]}}},
                "healthcheck": {"test": ["CMD-SHELL", "python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=5)\""], "interval": "10s", "timeout": "6s", "retries": 12, "start_period": "20m"},
            }
        }
    }
    compose_path = work / "compose.json"
    compose_path.write_text(json.dumps(compose, indent=2), encoding="utf-8")
    return compose_path, deploy_path


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 900) -> tuple[bytes, dict[str, str]]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"} if body else {}, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} at {url}: {detail}") from exc


def _vram_mib() -> float | None:
    probe = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, capture_output=True)
    try:
        return float(probe.stdout.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


def _wait_ready(compose: Path, config: dict[str, Any]) -> float:
    runtime = config["runtime"]
    port = int(os.environ.get("AEGIS_QWEN_COLD_PORT", runtime["port"]))
    base = f"http://127.0.0.1:{port}"
    started = time.perf_counter()
    deadline = started + int(runtime["ready_timeout_seconds"])
    while time.perf_counter() < deadline:
        try:
            data, _ = _http_json(base + "/v1/models", timeout=5)
            if EXPECTED_MODEL_PATH not in data.decode("utf-8", errors="replace"):
                raise RuntimeError("vLLM did not expose the expected 0.6B model")
            _http_json(base + "/health", timeout=5)
            return time.perf_counter() - started
        except (OSError, RuntimeError, urllib.error.URLError):
            if not _container_running(runtime["container_name"]):
                logs = subprocess.run(["docker", "logs", "--tail", "120", runtime["container_name"]], text=True, capture_output=True)
                raise RuntimeError("Qwen cold-lab container exited before readiness:\n" + logs.stdout + logs.stderr)
            time.sleep(5)
    raise TimeoutError(f"Qwen cold-lab did not become ready within {runtime['ready_timeout_seconds']} seconds")


def _source_path(cfg: Settings, profile: str, sentence_id: str) -> Path:
    return _paths(cfg)["root"] / "sources" / profile / f"{sentence_id}.wav"


def _source_cache_hit(cfg: Settings, record: dict[str, Any]) -> bool:
    path = Path(record.get("path", ""))
    return path.is_file() and record.get("status") == "complete" and record.get("sha256") == sha256(path)


def _post_speech(base: str, text: str, runtime: dict[str, Any]) -> tuple[bytes, float]:
    payload = {"model": EXPECTED_MODEL_PATH, "input": text, "task_type": "Base", "x_vector_only_mode": False, "language": "Portuguese", "response_format": "wav", "non_streaming_mode": True, "max_new_tokens": runtime["max_new_tokens"], "seed": runtime["seed"]}
    with Timer() as clock:
        audio, _ = _http_json(base + "/v1/audio/speech", payload)
    return audio, clock.seconds


def _generate_profile(cfg: Settings, config: dict[str, Any], code: str, keep_running: bool) -> None:
    runtime = config["runtime"]
    compose, deploy = _write_runtime_files(cfg, config, code)
    if _container_running(runtime["container_name"]):
        raise RuntimeError(f"refusing to alter an already-running {runtime['container_name']} container")
    subprocess.run(_compose_command(compose, config, "config"), check=True, stdout=subprocess.DEVNULL)
    try:
        subprocess.run(_compose_command(compose, config, "up", "--detach", "--no-build", runtime["service"]), check=True)
        ready_seconds = _wait_ready(compose, config)
        port = int(os.environ.get("AEGIS_QWEN_COLD_PORT", runtime["port"]))
        base = f"http://127.0.0.1:{port}"
        # Prompt construction and CUDA graphs are warmed before measured rows.
        _post_speech(base, "Todos os sistemas estão funcionando normalmente.", runtime)
        source_rows = read_jsonl(_paths(cfg)["sources"])
        existing = {(x.get("profile"), x.get("sentence_id")): x for x in source_rows}
        ref_audio, ref_text = _reference(config)
        corpus = yaml_config("corpus.yaml")["sentences"]
        for sentence in corpus:
            key = (code, sentence["id"])
            if key in existing and _source_cache_hit(cfg, existing[key]):
                continue
            path = _source_path(cfg, code, sentence["id"])
            try:
                payload, elapsed = _post_speech(base, sentence["text"], runtime)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                import soundfile as sf
                wave, sample_rate = sf.read(path, always_2d=False)
                wave = np.asarray(wave, dtype=float)
                if sample_rate != EXPECTED_SAMPLE_RATE:
                    raise ValueError(f"unexpected Qwen sample rate {sample_rate}; expected {EXPECTED_SAMPLE_RATE}")
                validate_audio(wave, sample_rate)
                append_jsonl(_paths(cfg)["sources"], {"engine": "qwen3-tts-vllm", "model": runtime["model"], "profile": code, "profile_description": config["profiles"][code]["description"], "stage0_sampling": {field: config["profiles"][code][field] for field in ("temperature", "top_p", "top_k", "repetition_penalty")}, "deploy_config_path": str(deploy), "sentence_id": sentence["id"], "text": sentence["text"], "path": str(path), "reference_wav": str(ref_audio), "reference_wav_sha256": sha256(ref_audio), "reference_transcript": str(ref_text), "reference_transcript_sha256": sha256(ref_text), "seed": runtime["seed"], "sample_rate": sample_rate, "duration_seconds": len(wave) / sample_rate, "cold_start_seconds": ready_seconds if sentence is corpus[0] else None, "warm_generation_seconds": elapsed, "rtf": elapsed / (len(wave) / sample_rate), "peak_vram_mib": _vram_mib(), "sha256": sha256(path), "status": "complete", "error": None})
            except Exception as exc:
                append_jsonl(_paths(cfg)["sources"], {"engine": "qwen3-tts-vllm", "model": runtime["model"], "profile": code, "sentence_id": sentence["id"], "text": sentence["text"], "status": "failed", "error": str(exc)})
    finally:
        if not keep_running:
            subprocess.run(_compose_command(compose, config, "down"), check=False)
    _process_effects(cfg, config, code)


def _append_candidate(cfg: Settings, source: dict[str, Any], engine: str, preset: str, raw: np.ndarray, final: np.ndarray, sample_rate: int, phases: dict[str, float], parameters: dict[str, Any]) -> None:
    import soundfile as sf
    raw_path = _paths(cfg)["root"] / "raw" / source["profile"] / preset / f'{source["sentence_id"]}.wav'
    mastered_path = _paths(cfg)["root"] / "mastered" / source["profile"] / preset / f'{source["sentence_id"]}.wav'
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    mastered_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(raw_path, raw, sample_rate)
    sf.write(mastered_path, final, sample_rate)
    duration = len(final) / sample_rate
    total = sum(phases.values())
    append_jsonl(_paths(cfg)["candidates"], {"source_profile": source["profile"], "engine": engine, "preset": preset, "sentence_id": source["sentence_id"], "text": source["text"], "source_path": source["path"], "output_raw_path": str(raw_path), "output_mastered_path": str(mastered_path), "parameters": parameters, "sample_rate": sample_rate, "input_samples": int(source["input_samples"]), "output_samples": len(final), "duration_seconds": duration, "analysis_seconds": phases["analysis"], "transform_seconds": phases["transform"], "synthesis_seconds": phases["synthesis"], "mastering_seconds": phases["mastering"], "total_dsp_seconds": total, "dsp_rtf": total / duration, "chatterbox_seconds": None, "qwen_generation_seconds": source["warm_generation_seconds"], "end_to_end_seconds": source["warm_generation_seconds"] + total, "peak_vram_mib": source.get("peak_vram_mib"), "sha256": sha256(mastered_path), "status": "complete", "error": None})


def _process_effects(cfg: Settings, config: dict[str, Any], profile: str) -> None:
    import soundfile as sf
    presets = yaml_config("presets.yaml")
    for source in read_jsonl(_paths(cfg)["sources"]):
        if source.get("profile") != profile or source.get("status") != "complete":
            continue
        source["input_samples"] = 0
        try:
            wave, sample_rate = sf.read(source["path"], always_2d=False)
            wave = np.asarray(wave, dtype=float)
            source["input_samples"] = len(wave)
            validate_audio(wave, sample_rate)
        except Exception as exc:
            for engine, preset in (("world", "W04"), ("lpc", "L00")):
                append_jsonl(_paths(cfg)["candidates"], {"source_profile": profile, "engine": engine, "preset": preset, "sentence_id": source.get("sentence_id"), "status": "failed", "error": str(exc)})
            continue
        try:
            with Timer() as analysis:
                world = analyze_world(wave, sample_rate, cfg.frame_period_ms)
            # W04: WORLD only sees analysis features and transformed features.
            w04 = presets["world"]["W04"]
            with Timer() as transform:
                f0 = transform_f0(world.f0_hz, world.voiced, **w04)
                envelope = transform_envelope(world.spectral_envelope, **w04)
                aperiodicity = transform_aperiodicity(world.aperiodicity, world.voiced, **w04)
            with Timer() as synthesis:
                raw = synthesize_world(world, f0_hz=f0, spectral_envelope=envelope, aperiodicity=aperiodicity, seed=cfg.seed, duration_samples=len(wave))
            with Timer() as mastering:
                final = master(raw, sample_rate, **presets["mastering"])
            validate_audio(raw, sample_rate, len(wave)); validate_audio(final, sample_rate, len(wave))
            _append_candidate(cfg, source, "world", "W04", raw, final, sample_rate, {"analysis": analysis.seconds, "transform": transform.seconds, "synthesis": synthesis.seconds, "mastering": mastering.seconds}, w04)
        except Exception as exc:
            append_jsonl(_paths(cfg)["candidates"], {"source_profile": profile, "engine": "world", "preset": "W04", "sentence_id": source.get("sentence_id"), "status": "failed", "error": str(exc)})
        try:
            with Timer() as analysis:
                world = analyze_world(wave, sample_rate, cfg.frame_period_ms)
                lpc = analyze_lpc(wave, sample_rate, world.f0_hz, world.voiced)
            l00 = presets["lpc"]["L00"]
            with Timer() as transform:
                lpc = replace(lpc, f0_hz=transform_f0(lpc.f0_hz, lpc.voiced, **l00))
            with Timer() as synthesis:
                raw = synthesize_lpc(lpc, seed=cfg.seed, duration_samples=len(wave), **l00)
                raw = np.pad(raw[:len(wave)], (0, max(0, len(wave) - len(raw))))
            with Timer() as mastering:
                final = master(raw, sample_rate, **presets["mastering"])
            validate_audio(raw, sample_rate, len(wave)); validate_audio(final, sample_rate, len(wave))
            _append_candidate(cfg, source, "lpc", "L00", raw, final, sample_rate, {"analysis": analysis.seconds, "transform": transform.seconds, "synthesis": synthesis.seconds, "mastering": mastering.seconds}, l00)
        except Exception as exc:
            append_jsonl(_paths(cfg)["candidates"], {"source_profile": profile, "engine": "lpc", "preset": "L00", "sentence_id": source.get("sentence_id"), "status": "failed", "error": str(exc)})


def _write_report(cfg: Settings) -> None:
    paths = _paths(cfg)
    sources = read_jsonl(paths["sources"])
    candidates = latest_candidates(read_jsonl(paths["candidates"]))
    completed = [row for row in candidates if row.get("status") == "complete"]
    latency: dict[str, Any] = {"sources": {}, "dsp": {}}
    for profile in sorted({row.get("profile") for row in sources if row.get("status") == "complete"}):
        rows = [row for row in sources if row.get("profile") == profile and row.get("status") == "complete"]
        latency["sources"][profile] = {"generation_seconds": stats([row["warm_generation_seconds"] for row in rows]), "rtf": stats([row["rtf"] for row in rows]), "peak_vram_mib": max((row.get("peak_vram_mib") or 0 for row in rows), default=0)}
    for key in sorted({(row.get("engine"), row.get("preset")) for row in completed}):
        rows = [row for row in completed if (row.get("engine"), row.get("preset")) == key]
        latency["dsp"][f"{key[0]}:{key[1]}"] = {"rtf": stats([row["dsp_rtf"] for row in rows]), "verdict": "acceptable" if np.median([row["dsp_rtf"] for row in rows]) <= .25 else "inadequate"}
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["latency"].write_text(json.dumps(latency, ensure_ascii=False, indent=2), encoding="utf-8")
    report = paths["reports"]
    report.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[str, Path]]] = {}
    for row in sources:
        if row.get("status") == "complete":
            grouped.setdefault(row["sentence_id"], []).append((f"Qwen {row['profile']} raw", Path(row["path"])))
    for row in completed:
        grouped.setdefault(row["sentence_id"], []).append((f"{row['source_profile']} {row['preset']}", Path(row["output_mastered_path"])))
    blocks = []
    for sentence, entries in grouped.items():
        audio = "".join(f'<p>{label}<br><audio controls src="../../../{path.resolve().relative_to(cfg.root.resolve()).as_posix()}"></audio></p>' for label, path in entries)
        blocks.append(f"<h2>{sentence}</h2>{audio}")
    (report / "listening.html").write_text("<meta charset=utf-8><h1>Qwen cold + strict resynthesis</h1>" + "\n".join(blocks), encoding="utf-8")
    (report / "manifest.json").write_text(json.dumps({"sources": sources, "candidates": candidates}, ensure_ascii=False, indent=2), encoding="utf-8")
    (report / "latency.html").write_text("<meta charset=utf-8><h1>Latency</h1><pre>" + json.dumps(latency, ensure_ascii=False, indent=2) + "</pre>", encoding="utf-8")
    (report / "summary.md").write_text("# Qwen cold laboratory\n\nNo automatic winner is selected. Q00–Q04 are explicit cold sampling profiles; each receives strict L00 and W04 resynthesis with no dry/source-audio mix.\n", encoding="utf-8")


def run(cfg: Settings, profiles: list[str] | None = None, keep_running: bool = False) -> None:
    config = _load_config()
    _check_prerequisites(cfg, config)
    wanted = profiles or list(config["profiles"])
    unknown = set(wanted) - set(config["profiles"])
    if unknown:
        raise ValueError("unknown Qwen cold profile(s): " + ", ".join(sorted(unknown)))
    if keep_running and len(wanted) != 1:
        raise ValueError("--keep-vllm requires exactly one profile; each profile has different deploy sampling")
    for code in wanted:
        _generate_profile(cfg, config, code, keep_running)
    _write_report(cfg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the isolated Qwen 0.6B cold-voice lab.")
    parser.add_argument("--profiles", help="comma-separated subset of Q00,Q01,Q02,Q03,Q04")
    parser.add_argument("--keep-vllm", action="store_true", help="leave only the lab-owned container running after the final profile")
    args = parser.parse_args(argv)
    from .config import settings
    run(settings(), args.profiles.split(",") if args.profiles else None, args.keep_vllm)
