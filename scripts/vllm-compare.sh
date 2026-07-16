#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/scripts/vllm_benchmark.py" compare \
  --reports "$ROOT/reports/vllm" \
  --samples "$ROOT/samples/benchmark-vllm" \
  --golden "${AEGIS_TTS_GOLDEN_1_7B:-}"
