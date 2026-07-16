#!/usr/bin/env python3
"""Create reproducible JSON/Markdown artifacts for the isolated 0.6B run.

This uses only Python's standard library so the host running the benchmark does
not need a second reporting environment or access to model dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
EXPECTED_VOICE = "giseli-34s"
EXPECTED_SEED = "41"


def _number(value: str, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path.name}: {field} não é numérico ({value!r})") from exc
    if result < 0:
        raise ValueError(f"{path.name}: {field} não pode ser negativo")
    return result


def _headers(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip().lower()] = value.strip()
    return data


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as source:
            if source.getcomptype() != "NONE":
                raise ValueError("WAV comprimido não é aceito")
            return source.getnframes() / source.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError) as exc:
        raise ValueError(f"{path.name}: WAV inválido ({exc})") from exc


def _mean_from_docker_stats(path: Path) -> float | None:
    values: list[float] = []
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        value = line.split(",", 1)[0].strip().rstrip("%")
        try:
            values.append(float(value))
        except ValueError:
            continue
    return statistics.fmean(values) if values else None


def _gpu_averages(path: Path) -> dict[str, float | None]:
    metrics: dict[str, list[float]] = {
        "sm": [],
        "power": [],
        "vram": [],
    }
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
            columns = [value.strip() for value in line.split(",")]
            if len(columns) != 4:
                continue
            for name, value in zip(("sm", "power", "vram"), columns[1:]):
                try:
                    metrics[name].append(float(value))
                except ValueError:
                    pass
    return {
        "sm_average_percent": statistics.fmean(metrics["sm"]) if metrics["sm"] else None,
        "power_average_watts": statistics.fmean(metrics["power"]) if metrics["power"] else None,
        "vram_peak_mib": max(metrics["vram"]) if metrics["vram"] else None,
    }


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _format(value: float | None, unit: str = "") -> str:
    return "N/D" if value is None else f"{value:.4f}{unit}"


def _read_health(path: Path) -> tuple[float | None, float | None]:
    if not path.is_file():
        return None, None
    try:
        health = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    load = health.get("load_seconds")
    prompts = health.get("prompt_seconds")
    prompt = prompts.get(EXPECTED_VOICE) if isinstance(prompts, dict) else None
    try:
        load_number = float(load) if load is not None else None
    except (TypeError, ValueError):
        load_number = None
    try:
        prompt_number = float(prompt) if prompt is not None else None
    except (TypeError, ValueError):
        prompt_number = None
    return load_number, prompt_number


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-vram-before-mib", type=float, default=None)
    parser.add_argument("--device-vram-after-mib", type=float, default=None)
    args = parser.parse_args()
    output_dir: Path = args.output_dir

    rows: list[dict[str, Any]] = []
    for index in range(1, 6):
        wav_path = output_dir / f"run-{index:02d}.wav"
        header_path = output_dir / f"run-{index:02d}.headers"
        if not wav_path.is_file() or not header_path.is_file():
            raise ValueError(f"artefato obrigatório ausente para run-{index:02d}")
        if not wav_path.read_bytes().startswith(b"RIFF"):
            raise ValueError(f"{wav_path.name}: resposta não é RIFF/WAV")
        headers = _headers(header_path)
        for key, expected in (
            ("x-aegis-tts-model", EXPECTED_MODEL),
            ("x-aegis-tts-voice", EXPECTED_VOICE),
            ("x-aegis-tts-seed", EXPECTED_SEED),
        ):
            if headers.get(key) != expected:
                raise ValueError(f"{header_path.name}: {key} esperado {expected!r}, obtido {headers.get(key)!r}")
        duration = _wav_duration(wav_path)
        reported_duration = _number(
            headers.get("x-aegis-tts-audio-duration-seconds", ""),
            "X-Aegis-TTS-Audio-Duration-Seconds",
            header_path,
        )
        if abs(duration - reported_duration) > 0.02:
            raise ValueError(
                f"{wav_path.name}: duração WAV ({duration:.4f}) diverge do header ({reported_duration:.4f})"
            )
        generation = _number(
            headers.get("x-aegis-tts-generation-seconds", ""),
            "X-Aegis-TTS-Generation-Seconds",
            header_path,
        )
        reported_rtf = _number(headers.get("x-aegis-tts-rtf", ""), "X-Aegis-TTS-RTF", header_path)
        calculated_rtf = generation / duration if duration else 0.0
        if abs(calculated_rtf - reported_rtf) > 0.02:
            raise ValueError(
                f"{header_path.name}: RTF calculado ({calculated_rtf:.4f}) diverge do header ({reported_rtf:.4f})"
            )
        rows.append(
            {
                "run": index,
                "wav": wav_path.name,
                "audio_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                "generation_seconds": generation,
                "audio_duration_seconds": duration,
                "rtf": calculated_rtf,
                "vram_before_mib": _number(
                    headers.get("x-aegis-tts-vram-before-mib", ""), "X-Aegis-TTS-VRAM-Before-MiB", header_path
                ),
                "vram_after_mib": _number(
                    headers.get("x-aegis-tts-vram-after-mib", ""), "X-Aegis-TTS-VRAM-After-MiB", header_path
                ),
                "vram_peak_mib": _number(
                    headers.get("x-aegis-tts-vram-peak-mib", ""), "X-Aegis-TTS-VRAM-Peak-MiB", header_path
                ),
            }
        )

    load_seconds, prompt_seconds = _read_health(output_dir / "health-ready.json")
    gpu = _gpu_averages(output_dir / "gpu-samples.csv")
    generation = [row["generation_seconds"] for row in rows]
    duration = [row["audio_duration_seconds"] for row in rows]
    rtf = [row["rtf"] for row in rows]
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": EXPECTED_MODEL,
        "voice": EXPECTED_VOICE,
        "seed": 41,
        "runs": rows,
        "audio_sha256": [row["audio_sha256"] for row in rows],
        "generation_seconds": [_round(value) for value in generation],
        "audio_duration_seconds": [_round(value) for value in duration],
        "rtf": [_round(value) for value in rtf],
        "median_generation_seconds": _round(statistics.median(generation)),
        "min_generation_seconds": _round(min(generation)),
        "max_generation_seconds": _round(max(generation)),
        "mean_generation_seconds": _round(statistics.fmean(generation)),
        "median_audio_duration_seconds": _round(statistics.median(duration)),
        "median_rtf": _round(statistics.median(rtf)),
        "min_rtf": _round(min(rtf)),
        "max_rtf": _round(max(rtf)),
        "mean_rtf": _round(statistics.fmean(rtf)),
        "vram_before_mib": _round(rows[0]["vram_before_mib"]),
        "vram_after_mib": _round(rows[-1]["vram_after_mib"]),
        "vram_peak_mib": _round(max(row["vram_peak_mib"] for row in rows)),
        "device_vram_before_mib": _round(args.device_vram_before_mib),
        "device_vram_after_mib": _round(args.device_vram_after_mib),
        "device_vram_peak_mib": _round(gpu["vram_peak_mib"]),
        "load_seconds": _round(load_seconds),
        "prompt_creation_seconds": _round(prompt_seconds),
        "container_cpu_average_percent": _round(_mean_from_docker_stats(output_dir / "docker-stats.csv")),
        "gpu_sm_average_percent": _round(gpu["sm_average_percent"]),
        "gpu_power_average_watts": _round(gpu["power_average_watts"]),
        "monitoring": {
            "docker_stats": "docker-stats.csv",
            "nvidia_smi_dmon": "nvidia-smi-dmon.log",
            "gpu_samples": "gpu-samples.csv",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Benchmark isolado — Qwen3-TTS 0.6B",
        "",
        f"Modelo: `{EXPECTED_MODEL}`  ",
        f"Voz: `{EXPECTED_VOICE}` · seed: `41` · 5 execuções medidas (warm-up excluído)",
        "",
        "| Run | Geração (s) | Duração WAV (s) | RTF | SHA-256 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        report.append(
            f"| {row['run']:02d} | {row['generation_seconds']:.4f} | {row['audio_duration_seconds']:.4f} | "
            f"{row['rtf']:.4f} | `{row['audio_sha256']}` |"
        )
    report.extend(
        [
            "",
            "## Resumo",
            "",
            f"- Tempo de carregamento: {_format(load_seconds, ' s')}",
            f"- Criação do prompt `giseli-34s`: {_format(prompt_seconds, ' s')}",
            f"- Geração — mediana/mínimo/máximo/média: {_format(statistics.median(generation), ' s')} / {_format(min(generation), ' s')} / {_format(max(generation), ' s')} / {_format(statistics.fmean(generation), ' s')}",
            f"- Duração de áudio mediana: {_format(statistics.median(duration), ' s')}",
            f"- RTF — mediana/mínimo/máximo/média: {_format(statistics.median(rtf))} / {_format(min(rtf))} / {_format(max(rtf))} / {_format(statistics.fmean(rtf))}",
            f"- VRAM do processo (headers) — antes/depois/pico: {_format(rows[0]['vram_before_mib'], ' MiB')} / {_format(rows[-1]['vram_after_mib'], ' MiB')} / {_format(max(row['vram_peak_mib'] for row in rows), ' MiB')}",
            f"- CPU média aproximada do container durante run-01: {_format(_mean_from_docker_stats(output_dir / 'docker-stats.csv'), ' %')}",
            f"- SM médio aproximado da GPU durante run-01: {_format(gpu['sm_average_percent'], ' %')}",
            f"- Potência média aproximada da GPU durante run-01: {_format(gpu['power_average_watts'], ' W')}",
            f"- VRAM global da GPU — antes/depois/pico monitorado: {_format(args.device_vram_before_mib, ' MiB')} / {_format(args.device_vram_after_mib, ' MiB')} / {_format(gpu['vram_peak_mib'], ' MiB')}",
            "",
            "Os valores globais da GPU podem incluir outros processos CUDA; os campos de VRAM dos headers pertencem ao processo de síntese. `nvidia-smi dmon -s pucm -d 1` foi preservado em `nvidia-smi-dmon.log`.",
            "",
            "A avaliação de qualidade é humana: os SHA-256 servem apenas para rastreabilidade e não constituem critério de qualidade.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(1)
