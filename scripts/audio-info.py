#!/usr/bin/env python3
"""Read-only PCM WAV metadata and basic signal measurements.

This small standard-library helper keeps the shell scripts usable on a host
that does not have FFmpeg installed.  It never converts, edits, or exports
audio.  It intentionally accepts normal uncompressed WAV/PCM files, including
the expected 32 kHz mono source format.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import wave
from pathlib import Path
from typing import Any, Iterator


SILENCE_THRESHOLD = 0.001
CLIPPING_THRESHOLD = 0.999


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def samples(raw: bytes, sample_width: int) -> Iterator[int]:
    if sample_width == 1:
        for item in raw:
            yield item - 128
    elif sample_width == 2:
        yield from (item[0] for item in struct.iter_unpack("<h", raw))
    elif sample_width == 3:
        for offset in range(0, len(raw) - 2, 3):
            value = raw[offset] | (raw[offset + 1] << 8) | (raw[offset + 2] << 16)
            if value & 0x800000:
                value -= 1 << 24
            yield value
    elif sample_width == 4:
        yield from (item[0] for item in struct.iter_unpack("<i", raw))
    else:
        raise ValueError(f"largura de amostra PCM não suportada: {sample_width} bytes")


def error_result(path: Path, message: str) -> dict[str, Any]:
    size = path.stat().st_size if path.exists() and path.is_file() else 0
    return {
        "path": str(path),
        "readable": False,
        "codec": "unknown",
        "sample_rate": None,
        "channels": None,
        "bits_per_sample": None,
        "duration_seconds": None,
        "size_bytes": size,
        "size_human": human_size(size),
        "peak": None,
        "rms": None,
        "silence_ratio": None,
        "clipping_ratio": None,
        "error": message,
    }


def inspect_pcm_wav(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return error_result(path, "arquivo ausente ou não é um arquivo regular")

    try:
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            compression = source.getcomptype()
            compression_name = source.getcompname()

            if compression != "NONE":
                return error_result(path, f"WAV comprimido não suportado: {compression_name}")
            if sample_rate <= 0 or channels <= 0:
                return error_result(path, "metadados WAV inválidos")
            if sample_width not in (1, 2, 3, 4):
                return error_result(path, f"bits por amostra não suportados: {sample_width * 8}")

            sample_total = 0
            square_total = 0.0
            max_abs = 0
            silent_total = 0
            clipped_total = 0
            full_scale = float(1 << (sample_width * 8 - 1))

            while True:
                raw = source.readframes(65_536)
                if not raw:
                    break
                for value in samples(raw, sample_width):
                    absolute = abs(value)
                    sample_total += 1
                    square_total += float(value) * float(value)
                    max_abs = max(max_abs, absolute)
                    normalized = absolute / full_scale
                    if normalized <= SILENCE_THRESHOLD:
                        silent_total += 1
                    if normalized >= CLIPPING_THRESHOLD:
                        clipped_total += 1

            duration = frame_count / sample_rate
            peak = max_abs / full_scale if sample_total else 0.0
            rms = math.sqrt(square_total / sample_total) / full_scale if sample_total else 0.0
            return {
                "path": str(path),
                "readable": True,
                "codec": "PCM",
                "sample_rate": sample_rate,
                "channels": channels,
                "bits_per_sample": sample_width * 8,
                "duration_seconds": duration,
                "size_bytes": path.stat().st_size,
                "size_human": human_size(path.stat().st_size),
                "peak": peak,
                "rms": rms,
                "silence_ratio": silent_total / sample_total if sample_total else 1.0,
                "clipping_ratio": clipped_total / sample_total if sample_total else 0.0,
                "error": None,
            }
    except (EOFError, OSError, wave.Error, ValueError, struct.error) as exc:
        return error_result(path, str(exc))


def as_tsv(result: dict[str, Any]) -> str:
    keys = (
        "codec",
        "sample_rate",
        "channels",
        "bits_per_sample",
        "duration_seconds",
        "size_bytes",
        "size_human",
        "peak",
        "rms",
        "silence_ratio",
        "clipping_ratio",
        "readable",
        "error",
    )
    fields: list[str] = []
    for key in keys:
        value = result.get(key)
        if value is None:
            fields.append("-")
        elif isinstance(value, float):
            fields.append(f"{value:.6f}")
        else:
            fields.append(str(value).replace("\t", " ").replace("\n", " "))
    return "\t".join(fields)


def as_human(result: dict[str, Any]) -> str:
    if not result["readable"]:
        return f"WAV inválido: {result['error']}"
    return "\n".join(
        (
            f"Codec: {result['codec']}",
            f"Sample rate: {result['sample_rate']} Hz",
            f"Canais: {result['channels']}",
            f"Bits por amostra: {result['bits_per_sample']}",
            f"Duração: {result['duration_seconds']:.3f} s",
            f"Tamanho: {result['size_human']}",
            f"Pico: {result['peak']:.6f}",
            f"RMS: {result['rms']:.6f}",
            f"Silêncio aproximado: {result['silence_ratio'] * 100:.2f}%",
            f"Clipping aproximado: {result['clipping_ratio'] * 100:.4f}%",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspeciona um WAV PCM sem modificá-lo")
    parser.add_argument("audio", type=Path)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_const", const="json", dest="format")
    output.add_argument("--tsv", action="store_const", const="tsv", dest="format")
    output.add_argument("--human", action="store_const", const="human", dest="format")
    parser.set_defaults(format="json")
    args = parser.parse_args()

    result = inspect_pcm_wav(args.audio)
    if args.format == "tsv":
        print(as_tsv(result))
    elif args.format == "human":
        print(as_human(result))
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["readable"] else 2


if __name__ == "__main__":
    sys.exit(main())
