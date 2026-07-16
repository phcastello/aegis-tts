#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE="${VLLM_COMPOSE:-$ROOT/experiments/vllm-omni/docker-compose.yml}"
SOURCE_CONFIG="$ROOT/experiments/vllm-omni/qwen3-tts-a2000.yaml"
RUNTIME_DIR="$ROOT/cache/vllm-omni/runtime-config"
UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.20}"
REPETITION_PENALTY="${VLLM_STAGE0_REPETITION_PENALTY:-1.05}"
MIN_TOKENS="${VLLM_STAGE0_MIN_TOKENS:-2}"

mkdir -p "$RUNTIME_DIR"
sed -E \
  -e "s/gpu_memory_utilization: 0\\.20/gpu_memory_utilization: ${UTILIZATION}/g" \
  -e "s/repetition_penalty: 1\\.05/repetition_penalty: ${REPETITION_PENALTY}/" \
  -e "s/min_tokens: 2/min_tokens: ${MIN_TOKENS}/" \
  "$SOURCE_CONFIG" >"$RUNTIME_DIR/qwen3-tts-runtime.yaml"

# Diagnostic-only switch. The primary benchmark leaves CUDA Graphs enabled.
if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
  sed -i -E \
    -e '0,/stage_id: 0/s/stage_id: 0/stage_id: 0\n    enforce_eager: true/' \
    -e 's/enforce_eager: false/enforce_eager: true/' \
    "$RUNTIME_DIR/qwen3-tts-runtime.yaml"
fi

docker compose -f "$COMPOSE" up -d --no-build aegis-tts-vllm
deadline=$((SECONDS + 1200))
until curl -fsS --max-time 5 http://127.0.0.1:8004/health >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    docker logs --tail 300 aegis-tts-vllm >&2
    exit 1
  fi
  if ! docker inspect -f '{{.State.Running}}' aegis-tts-vllm 2>/dev/null | grep -qx true; then
    docker logs --tail 300 aegis-tts-vllm >&2
    exit 1
  fi
  sleep 5
done
