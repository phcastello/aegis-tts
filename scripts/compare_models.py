#!/usr/bin/env python3
"""Compare a completed 0.6B benchmark with the fixed 1.7B baseline."""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINE = {
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "median_generation_seconds": 87.2500,
    "median_rtf": 5.2434,
    "audio_duration_seconds": 16.6400,
    # The supplied baseline does not contain comparable process-level VRAM.
    "vram_mib": None,
}


def _speed_classification(rtf: float) -> str:
    if rtf <= 1.0:
        return "candidato real para reprodução contínua"
    if rtf <= 2.0:
        return "candidato interativo com alguma latência"
    if rtf <= 3.5:
        return "melhoria material, mas ainda lento para diálogo"
    if rtf < 4.5:
        return "ganho moderado"
    return "ganho insuficiente para justificar troca"


def _value(value: float | None, unit: str = "") -> str:
    return "N/D" if value is None else f"{value:.4f}{unit}"


def _relative_audio_source(report_dir: Path, audio_path: Path) -> str:
    try:
        return audio_path.relative_to(report_dir).as_posix()
    except ValueError:
        return audio_path.resolve().as_uri()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--golden-1.7b", dest="golden_1_7b", type=Path)
    parser.add_argument("--baseline-vram-mib", type=float)
    args = parser.parse_args()
    try:
        candidate: dict[str, Any] = json.loads(args.summary.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"não foi possível ler summary 0.6B: {exc}") from exc
    if candidate.get("model") != "Qwen/Qwen3-TTS-12Hz-0.6B-Base":
        raise ValueError("summary não pertence ao modelo 0.6B esperado")
    for key in ("median_generation_seconds", "median_rtf", "median_audio_duration_seconds", "vram_peak_mib"):
        if not isinstance(candidate.get(key), (int, float)):
            raise ValueError(f"summary 0.6B não contém {key} numérico")

    baseline_vram = args.baseline_vram_mib
    candidate_rtf = float(candidate["median_rtf"])
    candidate_generation = float(candidate["median_generation_seconds"])
    candidate_duration = float(candidate["median_audio_duration_seconds"])
    candidate_vram = float(candidate["vram_peak_mib"])
    rtf_reduction = 100 * (BASELINE["median_rtf"] - candidate_rtf) / BASELINE["median_rtf"]
    generation_reduction = 100 * (BASELINE["median_generation_seconds"] - candidate_generation) / BASELINE["median_generation_seconds"]
    duration_difference = candidate_duration - BASELINE["audio_duration_seconds"]
    vram_difference = candidate_vram - baseline_vram if baseline_vram is not None else None

    comparison: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_1_7b": {**BASELINE, "vram_mib": baseline_vram},
        "candidate_0_6b": {
            "model": candidate["model"],
            "median_generation_seconds": candidate_generation,
            "median_rtf": candidate_rtf,
            "median_audio_duration_seconds": candidate_duration,
            "vram_peak_mib": candidate_vram,
        },
        "rtf_reduction_percent": round(rtf_reduction, 4),
        "generation_time_reduction_percent": round(generation_reduction, 4),
        "audio_duration_difference_seconds": round(duration_difference, 4),
        "vram_difference_mib": round(vram_difference, 4) if vram_difference is not None else None,
        "speed_classification": _speed_classification(candidate_rtf),
        "quality_assessment": "Pendente de avaliação auditiva humana; não inferida de RTF ou SHA-256.",
        "quality_checklist": [
            "identidade da Giseli",
            "sotaque brasileiro",
            "naturalidade",
            "ritmo",
            "entonação",
            "pronúncia",
            "estabilidade",
            "glitches",
            "repetição ou omissão de palavras",
            "diferenças em relação ao 1.7B",
        ],
    }
    reports_dir = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "compare-1.7b-vs-0.6b.json"
    markdown_path = reports_dir / "compare-1.7b-vs-0.6b.md"
    html_path = reports_dir / "compare-1.7b-vs-0.6b.html"
    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    vram_note = (
        "N/D — o baseline fornecido não registra VRAM de processo comparável. "
        "Informe `AEGIS_TTS_BASELINE_1_7B_VRAM_MIB` somente se tiver essa medição."
        if vram_difference is None
        else f"{vram_difference:+.4f} MiB (pico 0.6B menos VRAM de referência 1.7B)"
    )
    markdown = [
        "# Comparação — Qwen3-TTS 1.7B Base vs 0.6B Base",
        "",
        "| Métrica | 1.7B baseline | 0.6B medido | Diferença |",
        "| --- | ---: | ---: | ---: |",
        f"| Mediana de geração | {BASELINE['median_generation_seconds']:.4f} s | {candidate_generation:.4f} s | {generation_reduction:+.4f}% de redução |",
        f"| Mediana de RTF | {BASELINE['median_rtf']:.4f} | {candidate_rtf:.4f} | {rtf_reduction:+.4f}% de redução |",
        f"| Duração de áudio | {BASELINE['audio_duration_seconds']:.4f} s | {candidate_duration:.4f} s | {duration_difference:+.4f} s |",
        f"| VRAM | N/D | {candidate_vram:.4f} MiB (pico de processo) | {vram_note} |",
        "",
        f"Classificação de velocidade: **{_speed_classification(candidate_rtf)}**.",
        "",
        "Isto não conclui que a qualidade seja suficiente. A decisão exige audição humana dos WAVs e avaliação de identidade da Giseli, sotaque brasileiro, naturalidade, ritmo, entonação, pronúncia, estabilidade, glitches, repetição/omissão de palavras e diferenças frente ao 1.7B.",
        "",
        "Critério de decisão: promissor apenas com RTF mediano ≤ 2,0 e qualidade auditiva aceitável; RTF > 4,5 encerra este modelo como alternativa interativa. O 1.7B não deve ser substituído automaticamente.",
        "",
        f"Página auditiva local: [{html_path.name}]({html_path.name})",
    ]
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    golden_markup: str
    if args.golden_1_7b is not None:
        golden = args.golden_1_7b.expanduser().resolve()
        if not golden.is_file():
            raise ValueError(f"golden 1.7B informado não existe: {golden}")
        golden_markup = (
            "<h2>1.7B — golden fornecido</h2>"
            f"<audio controls preload=\"metadata\" src=\"{html.escape(_relative_audio_source(reports_dir, golden), quote=True)}\"></audio>"
            f"<p><code>{html.escape(str(golden))}</code></p>"
        )
    else:
        golden_markup = (
            "<h2>1.7B — golden</h2><p>Áudio golden não fornecido. Nenhum foi gerado automaticamente. "
            "Forneça um WAV existente com <code>AEGIS_TTS_GOLDEN_1_7B=/caminho/golden.wav "
            "./scripts/compare-models.sh</code> e abra esta página novamente.</p>"
        )
    sample_dir = args.summary.parent
    players = []
    for index in range(1, 6):
        wav = sample_dir / f"run-{index:02d}.wav"
        if not wav.is_file():
            raise ValueError(f"WAV 0.6B ausente: {wav}")
        source = html.escape(_relative_audio_source(reports_dir, wav), quote=True)
        players.append(
            f"<h2>0.6B — run {index:02d}</h2><audio controls preload=\"metadata\" src=\"{source}\"></audio>"
        )
    html_path.write_text(
        "<!doctype html><html lang=\"pt-BR\"><meta charset=\"utf-8\">"
        "<title>Comparação auditiva Qwen3-TTS</title>"
        "<style>body{font:16px system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem}audio{width:100%}code{overflow-wrap:anywhere}</style>"
        "<h1>Comparação auditiva — Qwen3-TTS</h1>"
        "<p>RTF e SHA-256 não são critérios de qualidade. Avalie identidade, sotaque, naturalidade, ritmo, entonação, pronúncia, estabilidade, glitches e palavras repetidas/omitidas.</p>"
        + golden_markup
        + "".join(players)
        + "</html>\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(1)
