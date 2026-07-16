#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE="$ROOT/docker-compose.eager-1.7b.yml"

docker compose -f "$COMPOSE" up -d --no-build aegis-tts
deadline=$((SECONDS + 900))
until curl -fsS --max-time 5 http://127.0.0.1:8001/health/ready >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    docker compose -f "$COMPOSE" logs --tail=200 aegis-tts >&2 || true
    exit 1
  fi
  sleep 5
done

echo "eager_1_7b_restored=true"
