#!/usr/bin/env bash
# Raw-Qwen diagnostic runner.  It never calls production port 8001 and never
# alters the protected Q00 YAML.  It does recreate the clean test container.
set -Eeuo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
compose_file="$project_dir/docker-compose.qwen-vllm-clean.yml"
profiles_dir="$project_dir/diagnostics/q01"
phrases_file="$profiles_dir/phrases.txt"
port=${AEGIS_QWEN_VLLM_CLEAN_PORT:-8004}
host=${AEGIS_Q01_HOST:-127.0.0.1}
stream=${AEGIS_Q01_HTTP_STREAM:-false}
seeds=${AEGIS_Q01_SEEDS:-"40733133 40733134 40733135 40733136 40733137"}
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
out_dir=${AEGIS_Q01_OUTPUT_DIR:-"$project_dir/diagnostics/q01/runs/$timestamp"}

if ! command -v jq >/dev/null; then
  echo 'jq is required to build safe JSON requests.' >&2
  exit 2
fi
if [[ "$stream" != true && "$stream" != false ]]; then
  echo 'AEGIS_Q01_HTTP_STREAM must be true or false.' >&2
  exit 2
fi

if (($#)); then
  profiles=("$@")
else
  profiles=(S00-q00-original S01-repetition-105 S02-topk-5 S03-topk-10 S04-topk-20 S05-official-defaults)
fi

mkdir -p "$out_dir"
printf 'profile\tphrase\tseed\tbytes\tduration_s\tfirst_sample_over_-40dBFS\tsha256\n' >"$out_dir/metrics.tsv"

wait_ready() {
  local attempts=0
  until curl --fail --silent --show-error "http://$host:$port/health" >/dev/null; do
    attempts=$((attempts + 1))
    if ((attempts >= 72)); then
      echo 'vLLM clean endpoint did not become ready within six minutes.' >&2
      return 1
    fi
    sleep 5
  done
}

for profile in "${profiles[@]}"; do
  deploy="/diagnostics/q01/$profile.yaml"
  [[ -f "$profiles_dir/$profile.yaml" ]] || { echo "Unknown profile: $profile" >&2; exit 2; }
  echo "== $profile =="
  (
    cd "$project_dir"
    AEGIS_VLLM_DEPLOY_CONFIG="$deploy" docker compose -f "$compose_file" up -d --force-recreate
  )
  wait_ready

  profile_dir="$out_dir/$profile"
  mkdir -p "$profile_dir"
  phrase_index=0
  while IFS= read -r phrase || [[ -n "$phrase" ]]; do
    [[ -n "$phrase" ]] || continue
    phrase_index=$((phrase_index + 1))
    for seed in $seeds; do
      stem=$(printf '%02d-seed-%s' "$phrase_index" "$seed")
      pcm="$profile_dir/$stem.pcm"
      body=$(jq -nc --arg input "$phrase" --argjson seed "$seed" --argjson stream "$stream" \
        '{input:$input,seed:$seed,stream:$stream,response_format:"pcm"} + (if $stream then {stream_format:"audio"} else {} end)')
      curl --no-buffer --fail --silent --show-error \
        "http://$host:$port/v1/audio/speech" \
        -H 'Content-Type: application/json' --data "$body" --output "$pcm"
      read -r bytes duration onset sha < <(python3 - "$pcm" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes()
if len(raw) % 2:
    raise SystemExit(f"odd PCM byte count: {len(raw)}")
samples = memoryview(raw).cast('h')
threshold = 655  # -40 dBFS rounded for PCM16 full scale
onset = next((i for i, sample in enumerate(samples) if abs(sample) >= threshold), -1)
print(len(raw), f"{len(samples) / 24000:.6f}", onset, hashlib.sha256(raw).hexdigest())
PY
)
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$profile" "$phrase_index" "$seed" "$bytes" "$duration" "$onset" "$sha" >>"$out_dir/metrics.tsv"
    done
  done <"$phrases_file"
done

echo "Raw PCM and metrics: $out_dir"
