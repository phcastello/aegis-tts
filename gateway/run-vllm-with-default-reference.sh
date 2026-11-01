#!/usr/bin/env bash
set -Eeuo pipefail

deploy_config="${AEGIS_VLLM_DEPLOY_CONFIG:-/protected/config/q00-vllm-deploy-exact.yaml}"

deploy_config="$(python3 /gateway/runtime_config.py --input "$deploy_config" \
  --output /tmp/aegis-runtime-deploy.yaml --codec-frames "${AEGIS_TTS_CODEC_BATCH_FRAMES:-25}")"

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
