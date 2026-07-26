#!/usr/bin/env bash
set -Eeuo pipefail

deploy_config="/protected/config/q00-vllm-deploy-exact.yaml"

vllm serve /models/Qwen3-TTS-12Hz-0.6B-Base \
  --deploy-config "$deploy_config" \
  --omni \
  --host 127.0.0.1 \
  --port 8091 \
  --trust-remote-code \
  --uvicorn-log-level warning &
backend_pid=$!

shutdown() {
  kill -TERM "$backend_pid" 2>/dev/null || true
  wait "$backend_pid" 2>/dev/null || true
}
trap shutdown EXIT INT TERM

python3 -m uvicorn default_reference_gateway:app --host 0.0.0.0 --port 8001 --log-level warning &
gateway_pid=$!

wait -n "$backend_pid" "$gateway_pid"
status=$?
kill -TERM "$gateway_pid" 2>/dev/null || true
wait "$gateway_pid" 2>/dev/null || true
exit "$status"
