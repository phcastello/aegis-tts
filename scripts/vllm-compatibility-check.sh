#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$ROOT/reports/vllm"
IMAGE="vllm/vllm-omni@sha256:8b0cc5438eb27b34cdfd22011b735da6a94835a09e9c56ddf9e8cb300d679919"
IMAGE_TAG="vllm/vllm-omni:v0.22.0"
OMNI_COMMIT="963ba1ab7a02b2c1f687285fcd2072cd57cd0b34"
mkdir -p "$REPORT_DIR"

gpu_csv="$(nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader,nounits)"
gpu_name="$(cut -d, -f1 <<<"$gpu_csv" | xargs)"
driver="$(cut -d, -f2 <<<"$gpu_csv" | xargs)"
compute_cap="$(cut -d, -f3 <<<"$gpu_csv" | xargs)"
memory_mib="$(cut -d, -f4 <<<"$gpu_csv" | xargs)"
host_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: \([^ ]*\).*/\1/p' | head -n1)"
docker_engine="$(docker version --format '{{.Server.Version}}')"
compose_version="$(docker compose version --short)"
toolkit_version="$(nvidia-container-cli --version | sed -n 's/^cli-version: //p')"
free_bytes="$(df --output=avail -B1 "$ROOT" | tail -n1 | xargs)"
compressed_bytes=9484794348
safe_required_bytes=20000000000

image_env="$(docker buildx imagetools inspect "$IMAGE_TAG" --format '{{json .Image}}')"
cuda_runtime="$(jq -r '."linux/amd64".config.Env[] | select(startswith("CUDA_VERSION=")) | split("=")[1]' <<<"$image_env")"
arch_list="$(jq -r '."linux/amd64".config.Env[] | select(startswith("TORCH_CUDA_ARCH_LIST=")) | split("=")[1]' <<<"$image_env")"
nvidia_requirement="$(jq -r '."linux/amd64".config.Env[] | select(startswith("NVIDIA_REQUIRE_CUDA=")) | sub("^NVIDIA_REQUIRE_CUDA="; "")' <<<"$image_env")"
image_present=false
runtime_probe="not_run"
runtime_probe_error=""
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  image_present=true
  set +e
  runtime_output="$(docker run --rm --gpus device=0 -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
    --entrypoint python3 "$IMAGE" -c \
    'import json,torch,vllm,vllm_omni; print(json.dumps({"torch":torch.__version__,"torch_cuda":torch.version.cuda,"vllm":vllm.__version__,"vllm_omni":vllm_omni.__version__,"cuda_available":torch.cuda.is_available(),"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"capability":torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None}))' 2>&1)"
  probe_rc=$?
  set -e
  if (( probe_rc == 0 )); then
    runtime_probe="passed"
  else
    runtime_probe="failed"
    runtime_probe_error="$runtime_output"
  fi
elif (( free_bytes < safe_required_bytes )); then
  runtime_probe="blocked_insufficient_disk"
  runtime_probe_error="Official image is 9484794348 compressed bytes; safe pull/extraction threshold is 20000000000 bytes, but only ${free_bytes} bytes are free."
else
  runtime_probe="image_not_pulled"
fi

jq -n \
  --arg generated_at "$(date -u +%FT%TZ)" \
  --arg gpu "$gpu_name" \
  --arg driver "$driver" \
  --arg host_cuda "$host_cuda" \
  --arg compute_capability "$compute_cap" \
  --arg architecture "NVIDIA Ampere GA106, compute capability 8.6" \
  --argjson gpu_memory_mib "$memory_mib" \
  --arg docker_engine "$docker_engine" \
  --arg docker_compose "$compose_version" \
  --arg nvidia_container_toolkit "$toolkit_version" \
  --arg vllm_version "0.22.0" \
  --arg vllm_omni_version "0.22.0" \
  --arg vllm_omni_commit "$OMNI_COMMIT" \
  --arg image "$IMAGE" \
  --arg image_tag "$IMAGE_TAG" \
  --arg cuda_runtime "$cuda_runtime" \
  --arg pytorch_version "2.11.0" \
  --arg torch_cuda_arch_list "$arch_list" \
  --arg nvidia_require_cuda "$nvidia_requirement" \
  --arg runtime_probe "$runtime_probe" \
  --arg runtime_probe_error "$runtime_probe_error" \
  --argjson image_present "$image_present" \
  --argjson image_compressed_bytes "$compressed_bytes" \
  --argjson filesystem_free_bytes "$free_bytes" \
  '{
    generated_at: $generated_at,
    host: {
      gpu: $gpu,
      gpu_architecture: $architecture,
      gpu_memory_mib: $gpu_memory_mib,
      driver_version: $driver,
      cuda_max_reported: $host_cuda,
      compute_capability: $compute_capability,
      docker_engine: $docker_engine,
      docker_compose: $docker_compose,
      nvidia_container_toolkit: $nvidia_container_toolkit
    },
    candidate: {
      vllm_version: $vllm_version,
      vllm_omni_version: $vllm_omni_version,
      vllm_omni_commit: $vllm_omni_commit,
      container_image: $image_tag,
      container_digest: $image,
      cuda_runtime: $cuda_runtime,
      pytorch_version: $pytorch_version,
      torch_cuda_arch_list: $torch_cuda_arch_list,
      nvidia_require_cuda: $nvidia_require_cuda
      ,api_schema_inspection: {
        source: "vllm_omni/entrypoints/openai/protocol/audio.py at v0.22.0",
        seed_supported: true,
        request_sampling_fields: ["max_new_tokens", "seed"],
        deploy_config_sampling_fields: ["temperature", "top_p", "top_k", "min_tokens", "repetition_penalty"],
        rejected_strategy: "Do not send undeclared temperature/top_p/top_k fields; pin them in Stage 0 deploy config."
      }
    },
    probe: {
      image_present: $image_present,
      image_compressed_bytes: $image_compressed_bytes,
      filesystem_free_bytes: $filesystem_free_bytes,
      status: $runtime_probe,
      error: $runtime_probe_error
    },
    fallback_candidate: {
      vllm_version: "0.20.0",
      vllm_omni_version: "0.20.0",
      container_image: "vllm/vllm-omni:v0.20.0",
      amd64_digest: "sha256:5d8c7e742c98858f257d82307e378391f0e7d77065e141c733cc4778042128ab",
      cuda_runtime: "13.0.2",
      pytorch_version: "2.11.0",
      note: "Only test if the primary image fails its real driver/runtime probe."
    }
  }' >"$REPORT_DIR/compatibility.json"

cat >"$REPORT_DIR/compatibility.md" <<EOF
# vLLM-Omni compatibility preflight

Generated: $(date -u +%FT%TZ)

## Host

- GPU: $gpu_name
- Architecture: NVIDIA Ampere GA106
- Compute capability: $compute_cap
- VRAM: $memory_mib MiB
- NVIDIA driver: $driver
- Maximum CUDA reported by host: $host_cuda
- Docker Engine: $docker_engine
- Docker Compose: $compose_version
- NVIDIA Container Toolkit: $toolkit_version

## Pinned primary candidate

- vLLM: 0.22.0
- vLLM-Omni: 0.22.0
- vLLM-Omni commit: $OMNI_COMMIT
- Image: \`$IMAGE_TAG\`
- amd64 digest: \`sha256:8b0cc5438eb27b34cdfd22011b735da6a94835a09e9c56ddf9e8cb300d679919\`
- CUDA runtime: $cuda_runtime
- PyTorch: 2.11.0
- Compiled CUDA architectures: $arch_list
- CUDA forward compatibility requested at runtime: yes
- Speech request schema: \`seed\` and \`max_new_tokens\` are accepted by
  \`/v1/audio/speech\`; temperature/top-p/top-k/min-tokens/repetition-penalty are
  pinned in Stage 0 because they are not declared request fields in 0.22.0.

## Real probe

- Status: **$runtime_probe**
- Image already present: $image_present
- Official image compressed size: $compressed_bytes bytes
- Filesystem free space: $free_bytes bytes
- Error/blocker: ${runtime_probe_error:-none}

The image metadata explicitly includes the 535 driver branch in
\`NVIDIA_REQUIRE_CUDA\`, but metadata is not counted as a successful runtime
test. A successful PyTorch CUDA initialization inside this exact digest is
required before the benchmark may be reported as executed.

## Official fallback (maximum second combination)

vLLM/vLLM-Omni 0.20.0, CUDA 13.0.2, PyTorch 2.11.0, amd64 digest
\`sha256:5d8c7e742c98858f257d82307e378391f0e7d77065e141c733cc4778042128ab\`.
It is attempted only if the primary exact image fails the real driver probe.
EOF

printf 'Wrote %s and %s\n' "$REPORT_DIR/compatibility.md" "$REPORT_DIR/compatibility.json"
