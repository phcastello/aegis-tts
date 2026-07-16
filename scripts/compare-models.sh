#!/usr/bin/env bash
# Compare only measured artifacts; this script never synthesizes or replaces
# either model and never makes a quality decision automatically.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SUMMARY="${PROJECT_ROOT}/samples/benchmark-0.6b/summary.json"
REPORTS_DIR="${PROJECT_ROOT}/reports"

[[ -f "$SUMMARY" ]] || {
  printf 'Erro: resumo 0.6B ausente; execute ./scripts/benchmark-0.6b.sh primeiro.\n' >&2
  exit 1
}

args=(
  "${SCRIPT_DIR}/compare_models.py"
  --summary "$SUMMARY"
  --reports-dir "$REPORTS_DIR"
)
if [[ -n "${AEGIS_TTS_GOLDEN_1_7B:-}" ]]; then
  args+=(--golden-1.7b "$AEGIS_TTS_GOLDEN_1_7B")
fi
if [[ -n "${AEGIS_TTS_BASELINE_1_7B_VRAM_MIB:-}" ]]; then
  args+=(--baseline-vram-mib "$AEGIS_TTS_BASELINE_1_7B_VRAM_MIB")
fi

python3 "${args[@]}"
printf 'Relatórios de comparação: %s/compare-1.7b-vs-0.6b.{json,md,html}\n' "$REPORTS_DIR"
