#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$ROOT/reports/vllm"
SAMPLE_DIR="$ROOT/samples/benchmark-vllm"
IMAGE="vllm/vllm-omni@sha256:8b0cc5438eb27b34cdfd22011b735da6a94835a09e9c56ddf9e8cb300d679919"
PRODUCTION_STOPPED=false
PRODUCTION_WAS_RUNNING=false
DMON_PID=""
STATS_PID=""
mkdir -p "$REPORT_DIR" "$SAMPLE_DIR/nonstream" "$SAMPLE_DIR/stream"

restore() {
  rc=$?
  trap - EXIT INT TERM
  if [[ -n "$DMON_PID" ]]; then
    kill "$DMON_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$STATS_PID" ]]; then
    kill "$STATS_PID" >/dev/null 2>&1 || true
  fi
  "$ROOT/scripts/vllm-stop.sh" || true
  if [[ "$PRODUCTION_WAS_RUNNING" == true ]]; then
    "$ROOT/scripts/vllm-restore-production.sh" || rc=1
  else
    {
      date -u +%FT%TZ
      echo "production_service_restored=not_applicable_was_not_running"
    } >"$REPORT_DIR/production-state-final.txt"
  fi
  exit "$rc"
}
trap restore EXIT INT TERM

"$ROOT/scripts/vllm-compatibility-check.sh"

{
  date -u +%FT%TZ
  echo "health_ready:"
  curl -fsS --max-time 5 http://127.0.0.1:8001/health/ready
  echo
  echo "docker_compose_ps:"
  docker compose -f "$ROOT/docker-compose.yml" ps
  echo "containers:"
  docker ps -a --no-trunc
  echo "production_mounts:"
  docker inspect aegis-tts --format '{{json .Mounts}}'
} >"$REPORT_DIR/production-state-before.txt"

if docker inspect -f '{{.State.Running}}' aegis-tts 2>/dev/null | grep -qx true; then
  PRODUCTION_WAS_RUNNING=true
fi

free_bytes="$(df --output=avail -B1 "$ROOT" | tail -n1 | xargs)"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if (( free_bytes < 20000000000 )); then
    other_gpu_names="$(docker ps --format '{{.Names}}' | while read -r name; do
      if [[ "$name" != "aegis-tts" ]] && docker inspect "$name" --format '{{json .HostConfig.DeviceRequests}}' | grep -q '"gpu"'; then
        echo "$name"
      fi
    done | paste -sd, -)"
    cat >"$REPORT_DIR/memory-fit.md" <<EOF
# A2000 memory-fit attempt

Status: **blocked before image pull**

- Official image compressed size: 9,484,794,348 bytes
- Filesystem free space: $free_bytes bytes
- Required safe pull/extraction headroom: 20,000,000,000 bytes
- Other running GPU containers not authorized to stop: ${other_gpu_names:-none}
- Production service stopped: false
- GPU memory configurations tested: none

No data was deleted and no Docker prune command was used.
EOF
    jq -n --argjson free "$free_bytes" --arg containers "$other_gpu_names" '{
      status: "blocked_insufficient_disk",
      filesystem_free_bytes: $free,
      required_safe_bytes: 20000000000,
      other_gpu_containers: (if $containers == "" then [] else ($containers | split(",")) end),
      attempts: [],
      production_service_stopped: false
    }' >"$REPORT_DIR/memory-fit.json"
    exit 2
  fi
  docker pull "$IMAGE"
fi

set +e
probe_output="$(docker run --rm --gpus device=0 -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  --entrypoint python3 "$IMAGE" -c \
  'import json,torch,vllm,vllm_omni; assert torch.cuda.is_available(); print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"vllm":vllm.__version__,"vllm_omni":vllm_omni.__version__,"gpu":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0)}))' 2>&1)"
probe_rc=$?
set -e
if (( probe_rc != 0 )); then
  printf '%s\n' "$probe_output" >"$REPORT_DIR/runtime-probe-error.txt"
  exit 3
fi
printf '%s\n' "$probe_output" >"$REPORT_DIR/runtime-probe.json"

other_gpu_containers=()
while IFS= read -r name; do
  [[ -z "$name" || "$name" == "aegis-tts" ]] && continue
  other_gpu_containers+=("$name")
done < <(docker ps --format '{{.Names}}' | while read -r name; do
  if docker inspect "$name" --format '{{json .HostConfig.DeviceRequests}}' | grep -q '"gpu"'; then
    echo "$name"
  fi
done)
if (( ${#other_gpu_containers[@]} > 0 )); then
  {
    echo "# A2000 memory-fit attempt"
    echo
    echo "Status: **blocked by another running GPU container**"
    echo
    for container in "${other_gpu_containers[@]}"; do
      printf '%s\n' "- Unauthorized-to-stop GPU container: \`$container\`"
    done
    echo "- Production service stopped: false"
    echo "- GPU memory configurations tested: none"
    echo
    echo "The benchmark script is permitted to stop only the aegis-tts container."
  } >"$REPORT_DIR/memory-fit.md"
  jq -n --arg containers "$(IFS=,; echo "${other_gpu_containers[*]}")" '{
    status: "blocked_other_gpu_container",
    other_gpu_containers: ($containers | split(",")),
    attempts: [],
    production_service_stopped: false
  }' >"$REPORT_DIR/memory-fit.json"
  exit 4
fi

if [[ "$PRODUCTION_WAS_RUNNING" == true ]]; then
  docker stop aegis-tts >/dev/null
  PRODUCTION_STOPPED=true
  date -u +%FT%TZ >"$REPORT_DIR/production-stop-window-start.txt"
fi

attempts_json='[]'
selected=""
# 0.20 was the conservative first value. vLLM reported it left no KV-cache
# blocks, so test a small set of still-reduced reservations before descending.
for utilization in 0.40 0.30 0.20 0.18 0.16 0.14 0.12 0.10; do
  before="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | xargs)"
  start_ns="$(date +%s%N)"
  set +e
  VLLM_GPU_MEMORY_UTILIZATION="$utilization" "$ROOT/scripts/vllm-start.sh" \
    >"$REPORT_DIR/start-${utilization}.log" 2>&1
  start_rc=$?
  if (( start_rc == 0 )); then
    python3 "$ROOT/scripts/vllm_benchmark.py" probe \
      --root "$ROOT" --output "$ROOT/cache/vllm-omni/fit-probe.wav" \
      >>"$REPORT_DIR/start-${utilization}.log" 2>&1
    synth_rc=$?
  else
    synth_rc=$start_rc
  fi
  set -e
  elapsed="$(python3 -c "print(($(date +%s%N)-$start_ns)/1e9)")"
  after="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | xargs)"
  kv_cache="$(docker logs aegis-tts-vllm 2>&1 | rg -i 'kv cache|gpu kv' | tail -n 4 || true)"
  error="$(docker logs aegis-tts-vllm 2>&1 | rg -i 'out of memory|oom|error|failed' | tail -n 12 || true)"
  success=false
  if (( synth_rc == 0 )); then
    success=true
    selected="$utilization"
  fi
  attempts_json="$(jq \
    --arg u "$utilization" --argjson before "$before" --argjson after "$after" \
    --argjson startup "$elapsed" --argjson success "$success" \
    --arg kv "$kv_cache" --arg error "$error" \
    '. + [{stage_0_utilization:($u|tonumber),stage_1_utilization:($u|tonumber),vram_before_mib:$before,vram_after_load_mib:$after,startup_seconds:$startup,success:$success,kv_cache:$kv,error:$error}]' \
    <<<"$attempts_json")"
  if [[ "$success" == true ]]; then
    break
  fi
  "$ROOT/scripts/vllm-stop.sh"
done

jq -n --arg selected "$selected" --argjson attempts "$attempts_json" '{
  status: (if $selected == "" then "no_fit" else "fit" end),
  selected_utilization: (if $selected == "" then null else ($selected|tonumber) end),
  attempts: $attempts,
  production_service_stopped: true
}' >"$REPORT_DIR/memory-fit.json"
python3 "$ROOT/scripts/vllm_benchmark.py" memory-report --input "$REPORT_DIR/memory-fit.json" \
  --output "$REPORT_DIR/memory-fit.md"
[[ -n "$selected" ]] || exit 5

nvidia-smi dmon -s pucvmet -d 1 -o DT >"$REPORT_DIR/nvidia-dmon.txt" 2>&1 &
DMON_PID=$!
(while true; do
  printf '%s,' "$(date -u +%FT%TZ)"
  docker stats --no-stream --format '{{.CPUPerc}},{{.MemUsage}}' aegis-tts-vllm 2>/dev/null || true
  sleep 1
done) >"$REPORT_DIR/docker-stats.csv" &
STATS_PID=$!

python3 "$ROOT/scripts/vllm_benchmark.py" run --root "$ROOT"

kill "$DMON_PID" "$STATS_PID" >/dev/null 2>&1 || true
DMON_PID=""
STATS_PID=""
python3 "$ROOT/scripts/vllm_benchmark.py" metrics --reports "$REPORT_DIR"
"$ROOT/scripts/vllm-compare.sh"

if [[ "$PRODUCTION_STOPPED" == true ]]; then
  date -u +%FT%TZ >"$REPORT_DIR/production-stop-window-end.txt"
fi
