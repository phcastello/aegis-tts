"""Static checks for the isolated experiment; no Docker/GPU/model is needed."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_vllm_0_6b_is_the_production_default_with_an_eager_rollback() -> None:
    production = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    rollback = (ROOT / "docker-compose.eager-1.7b.yml").read_text(encoding="utf-8")

    assert "container_name: aegis-tts\n" in production
    assert "vllm/vllm-omni@sha256:8b0cc5438eb27b34cdfd22011b735da6a94835a09e9c56ddf9e8cb300d679919" in production
    assert "/models/Qwen3-TTS-12Hz-0.6B-Base" in production
    assert "/config/vllm-qwen3-tts-0.6b.yaml" in production
    assert "./giselle_AegisVoice3.wav:/defaults/reference.wav:ro" in production
    assert "./giselle_AegisVoice3.txt:/defaults/reference.txt:ro" in production
    assert "/gateway/run-vllm-with-default-reference.sh" in production
    assert "${AEGIS_TTS_PORT:-8001}" in production
    assert "Qwen3-TTS-12Hz-1.7B-Base" not in production

    assert "image: aegis-tts:local" in rollback
    assert "QWEN_MODEL_ID: Qwen/Qwen3-TTS-12Hz-1.7B-Base" in rollback
    assert "QWEN_MODEL_DIR: /models/Qwen3-TTS-12Hz-1.7B-Base" in rollback


def test_0_6b_scripts_are_explicit_and_do_not_target_the_production_service() -> None:
    downloader = (ROOT / "scripts" / "download-model-0.6b.sh").read_text(encoding="utf-8")
    benchmark = (ROOT / "scripts" / "benchmark-0.6b.sh").read_text(encoding="utf-8")

    assert 'MODEL_ID="Qwen/Qwen3-TTS-12Hz-0.6B-Base"' in downloader
    assert 'MODEL_DIR_HOST="${PROJECT_ROOT}/models/Qwen3-TTS-12Hz-0.6B-Base"' in downloader
    assert "aegis-tts-0.6b download \"$MODEL_ID\"" in downloader
    assert "docker system prune" not in downloader
    assert 'SERVICE="aegis-tts-0.6b"' in benchmark
    assert 'BASE_URL="http://127.0.0.1:8003"' in benchmark
    assert "docker compose -f \"$COMPOSE_FILE\" up -d --no-deps \"$SERVICE\"" in benchmark
    assert "docker stop aegis-tts" not in benchmark
    assert "docker restart aegis-tts" not in benchmark


def test_comparison_report_works_without_inventing_a_1_7b_golden(tmp_path: Path) -> None:
    """The report is offline and must not generate a golden audio by itself."""

    samples = tmp_path / "samples" / "benchmark-0.6b"
    samples.mkdir(parents=True)
    summary = {
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "median_generation_seconds": 42.0,
        "median_rtf": 2.5,
        "median_audio_duration_seconds": 16.64,
        "vram_peak_mib": 2048.0,
    }
    (samples / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for index in range(1, 6):
        (samples / f"run-{index:02d}.wav").write_bytes(b"test-only-placeholder")

    reports = tmp_path / "reports"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_models.py"),
            "--summary",
            str(samples / "summary.json"),
            "--reports-dir",
            str(reports),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    comparison = json.loads((reports / "compare-1.7b-vs-0.6b.json").read_text(encoding="utf-8"))
    assert comparison["quality_assessment"].startswith("Pendente")
    page = (reports / "compare-1.7b-vs-0.6b.html").read_text(encoding="utf-8")
    assert "Áudio golden não fornecido" in page
    assert "0.6B — run 05" in page
