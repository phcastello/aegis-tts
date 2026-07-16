#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_LOG="$ROOT/reports/vllm/production-state-final.txt"
mkdir -p "$(dirname "$STATE_LOG")"

docker start aegis-tts >/dev/null
deadline=$((SECONDS + 1200))
until curl -fsS --max-time 5 http://127.0.0.1:8001/health >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    {
      date -u +%FT%TZ
      echo "production_service_restored=false"
      docker ps -a --filter name='^/aegis-tts$' --no-trunc
      docker logs --tail 100 aegis-tts 2>&1
    } >"$STATE_LOG"
    exit 1
  fi
  sleep 5
done

{
  date -u +%FT%TZ
  echo "production_service_restored=true"
  curl -fsS --max-time 5 http://127.0.0.1:8001/health
  docker ps -a --filter name='^/aegis-tts$' --no-trunc
} >"$STATE_LOG"
