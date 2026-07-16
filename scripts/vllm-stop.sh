#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE="$ROOT/experiments/vllm-omni/docker-compose.yml"
docker compose -f "$COMPOSE" stop aegis-tts-vllm >/dev/null 2>&1 || true
