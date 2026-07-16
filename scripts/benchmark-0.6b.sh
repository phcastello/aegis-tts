#!/usr/bin/env bash
# Reproducible, sequential benchmark for the isolated 0.6B service. It never
# calls docker stop/restart on the production aegis-tts service at port 8001.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.0.6b.yml"
SERVICE="aegis-tts-0.6b"
CONTAINER="aegis-tts-0.6b"
BASE_URL="http://127.0.0.1:8003"
MODEL_ID="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_DIR="${PROJECT_ROOT}/models/Qwen3-TTS-12Hz-0.6B-Base"
VOICE_ID="giseli-34s"
OUTPUT_DIR="${PROJECT_ROOT}/samples/benchmark-0.6b"
READY_TIMEOUT_SECONDS="${AEGIS_TTS_READY_TIMEOUT_SECONDS:-900}"

die() {
  printf 'Erro: %s\n' "$*" >&2
  exit 1
}

require_checkpoint() {
  [[ -f "${MODEL_DIR}/config.json" ]] || die "modelo 0.6B ausente; execute ./scripts/download-model-0.6b.sh"
  python3 - "$MODEL_DIR" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
if (directory / "model.safetensors").is_file():
    raise SystemExit(0)
index = directory / "model.safetensors.index.json"
if not index.is_file():
    raise SystemExit("pesos principais 0.6B ausentes")
try:
    files = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
except (OSError, ValueError, KeyError, AttributeError) as exc:
    raise SystemExit(f"índice de pesos 0.6B inválido: {exc}")
if not files or any(not (directory / name).is_file() for name in files):
    raise SystemExit("pesos fragmentados 0.6B ausentes")
PY
}

require_reference() {
  python3 - "$PROJECT_ROOT/config/voices.json" "$PROJECT_ROOT/voices" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

config_path, voices_dir = map(Path, sys.argv[1:])
try:
    document = json.loads(config_path.read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    raise SystemExit(f"configuração de vozes inválida: {exc}")
voice = next((item for item in document.get("voices", []) if item.get("id") == "giseli-34s"), None)
if not voice or not voice.get("enabled", False):
    raise SystemExit("referência giseli-34s não está habilitada")
if voice.get("audio_path") != "/voices/giseli-34s/reference.wav" or voice.get("transcript_path") != "/voices/giseli-34s/reference.txt":
    raise SystemExit("giseli-34s não aponta para a referência aprovada esperada")
for suffix in ("reference.wav", "reference.txt"):
    path = voices_dir / "giseli-34s" / suffix
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"referência ausente ou vazia: {path}")
PY
}

ARTIFACTS=(
  benchmark-request.json compose-config.yml health-ready.json info.json warmup.wav warmup.headers
  run-01.wav run-01.headers run-02.wav run-02.headers run-03.wav run-03.headers
  run-04.wav run-04.headers run-05.wav run-05.headers
  docker-stats.csv nvidia-smi-dmon.log gpu-samples.csv
  summary.json summary.md
)

prepare_output_dir() {
  mkdir -p "$OUTPUT_DIR"
  local present=0 artifact
  for artifact in "${ARTIFACTS[@]}"; do
    [[ -e "${OUTPUT_DIR}/${artifact}" ]] && present=1
  done
  (( present == 0 )) && return 0
  if [[ "${AEGIS_TTS_BENCHMARK_OVERWRITE:-}" != "1" ]]; then
    die "já há artefatos de benchmark em ${OUTPUT_DIR}; use AEGIS_TTS_BENCHMARK_OVERWRITE=1 para arquivá-los conscientemente antes de repetir"
  fi
  local archive_dir="${OUTPUT_DIR}/history/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$archive_dir"
  for artifact in "${ARTIFACTS[@]}"; do
    [[ -e "${OUTPUT_DIR}/${artifact}" ]] && mv "${OUTPUT_DIR}/${artifact}" "$archive_dir/"
  done
  printf 'Artefatos anteriores preservados em %s\n' "$archive_dir"
}

vram_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {gsub(/ /, ""); print; exit}'
}

DOCKER_MONITOR_PID=""
GPU_MONITOR_PID=""
DMON_PID=""

stop_monitors() {
  local pid
  for pid in "$DOCKER_MONITOR_PID" "$GPU_MONITOR_PID" "$DMON_PID"; do
    [[ -n "$pid" ]] || continue
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  DOCKER_MONITOR_PID=""
  GPU_MONITOR_PID=""
  DMON_PID=""
}

trap stop_monitors EXIT

monitor_docker_stats() {
  local curl_pid="$1"
  while kill -0 "$curl_pid" 2>/dev/null; do
    docker stats --no-stream --format '{{.CPUPerc}},{{.MemUsage}}' "$CONTAINER" 2>/dev/null || true
    sleep 1
  done
}

monitor_gpu_samples() {
  local curl_pid="$1"
  while kill -0 "$curl_pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk 'NR == 1 {gsub(/ /, ""); printf "%s,%s\n", strftime("%Y-%m-%dT%H:%M:%SZ", systime()), $0; exit}' || true
    sleep 1
  done
}

validate_response() {
  local wav="$1" headers="$2"
  python3 - "$wav" "$headers" "$MODEL_ID" "$VOICE_ID" <<'PY'
from __future__ import annotations
import sys
import wave
from pathlib import Path

wav_path = Path(sys.argv[1])
headers_path = Path(sys.argv[2])
expected_model = sys.argv[3]
expected_voice = sys.argv[4]
if not wav_path.read_bytes().startswith(b"RIFF"):
    raise SystemExit(f"{wav_path.name}: resposta não começa com RIFF")
try:
    with wave.open(str(wav_path), "rb") as source:
        if source.getnchannels() != 1 or source.getframerate() <= 0 or source.getnframes() <= 0:
            raise SystemExit(f"{wav_path.name}: WAV inválido")
except (wave.Error, EOFError) as exc:
    raise SystemExit(f"{wav_path.name}: WAV inválido ({exc})")
headers = {}
for line in headers_path.read_text(encoding="utf-8").splitlines():
    if ":" in line:
        key, value = line.split(":", 1)
        headers[key.lower().strip()] = value.strip()
required = {
    "x-aegis-tts-model", "x-aegis-tts-voice", "x-aegis-tts-seed",
    "x-aegis-tts-generation-seconds", "x-aegis-tts-audio-duration-seconds",
    "x-aegis-tts-rtf", "x-aegis-tts-queue-wait-seconds",
    "x-aegis-tts-sample-rate", "x-aegis-tts-dtype",
    "x-aegis-tts-vram-before-mib", "x-aegis-tts-vram-after-mib",
    "x-aegis-tts-vram-peak-mib",
}
missing = sorted(required - set(headers))
if missing:
    raise SystemExit("headers X-Aegis-TTS ausentes: " + ", ".join(missing))
if headers["x-aegis-tts-model"] != expected_model:
    raise SystemExit(f"modelo retornado inesperado: {headers['x-aegis-tts-model']!r}")
if headers["x-aegis-tts-voice"] != expected_voice:
    raise SystemExit(f"voz retornada inesperada: {headers['x-aegis-tts-voice']!r}")
if headers["x-aegis-tts-seed"] != "41":
    raise SystemExit(f"seed retornado inesperado: {headers['x-aegis-tts-seed']!r}")
PY
}

generate_audio() {
  local label="$1" monitored="$2"
  local wav_path="${OUTPUT_DIR}/${label}.wav"
  local headers_path="${OUTPUT_DIR}/${label}.headers"
  local raw_headers="${OUTPUT_DIR}/.${label}.raw-headers"
  local curl_pid
  local -a curl_args=(
    --silent --show-error --fail --connect-timeout 5 --max-time 300
    -X POST "${BASE_URL}/v1/synthesize"
    -H 'Content-Type: application/json'
    --data-binary "@${OUTPUT_DIR}/benchmark-request.json"
  )
  if [[ -n "${AEGIS_TTS_API_KEY:-}" ]]; then
    curl_args+=(-H "X-API-Key: ${AEGIS_TTS_API_KEY}")
  fi

  curl "${curl_args[@]}" -D "$raw_headers" -o "$wav_path" &
  curl_pid=$!
  if [[ "$monitored" == "yes" ]]; then
    : > "${OUTPUT_DIR}/docker-stats.csv"
    printf 'timestamp,utilization_gpu_percent,power_watts,memory_used_mib\n' > "${OUTPUT_DIR}/gpu-samples.csv"
    monitor_docker_stats "$curl_pid" >> "${OUTPUT_DIR}/docker-stats.csv" &
    DOCKER_MONITOR_PID=$!
    monitor_gpu_samples "$curl_pid" >> "${OUTPUT_DIR}/gpu-samples.csv" &
    GPU_MONITOR_PID=$!
    nvidia-smi dmon -s pucm -d 1 > "${OUTPUT_DIR}/nvidia-smi-dmon.log" 2>&1 &
    DMON_PID=$!
  fi
  if ! wait "$curl_pid"; then
    stop_monitors
    [[ -s "$raw_headers" ]] && sed -n '1,80p' "$raw_headers" >&2 || true
    die "falha HTTP durante ${label}"
  fi
  [[ "$monitored" == "yes" ]] && stop_monitors
  if ! grep -qi '^content-type:[[:space:]]*audio/wav' "$raw_headers"; then
    sed -n '1,80p' "$raw_headers" >&2 || true
    die "${label} não retornou Content-Type audio/wav"
  fi
  grep -i '^X-Aegis-TTS-' "$raw_headers" | tr -d '\r' > "$headers_path"
  rm -f "$raw_headers"
  validate_response "$wav_path" "$headers_path"
}

wait_for_ready() {
  local temporary="${OUTPUT_DIR}/.health-ready.tmp"
  local status elapsed=0
  while (( elapsed < READY_TIMEOUT_SECONDS )); do
    status="$(curl --silent --show-error --max-time 5 --output "$temporary" --write-out '%{http_code}' "${BASE_URL}/health/ready" || true)"
    if [[ "$status" == "200" ]]; then
      python3 - "$temporary" "$MODEL_ID" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

health = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if health.get("status") != "ready" or health.get("model") != sys.argv[2]:
    raise SystemExit("/health/ready não confirmou o modelo 0.6B pronto")
PY
      mv "$temporary" "${OUTPUT_DIR}/health-ready.json"
      return 0
    fi
    sleep 1
    ((elapsed += 1))
  done
  rm -f "$temporary"
  die "o serviço experimental não ficou pronto em ${READY_TIMEOUT_SECONDS}s; consulte docker logs ${CONTAINER}"
}

cd "$PROJECT_ROOT"
command -v docker >/dev/null 2>&1 || die 'Docker não encontrado.'
command -v curl >/dev/null 2>&1 || die 'curl não encontrado.'
command -v nvidia-smi >/dev/null 2>&1 || die 'nvidia-smi não encontrado; este benchmark exige GPU NVIDIA.'
docker compose version >/dev/null 2>&1 || die 'Docker Compose não encontrado.'
nvidia-smi -L >/dev/null || die 'nenhuma GPU NVIDIA acessível pelo host.'
[[ -f "$COMPOSE_FILE" ]] || die "Compose experimental ausente: ${COMPOSE_FILE}"

require_checkpoint
require_reference
prepare_output_dir

cat > "${OUTPUT_DIR}/benchmark-request.json" <<'JSON'
{
  "text": "Olá, Pedro. Pelo que eu verifiquei, todos os serviços estão funcionando normalmente. A reunião continua marcada para segunda-feira, e ainda há algumas mensagens que precisam de resposta. Depois disso, posso organizar as próximas tarefas e revisar o que ficou pendente para amanhã.",
  "voice": "giseli-34s",
  "language": "pt-BR",
  "seed": 41,
  "generation_parameters": {
    "temperature": 0.8,
    "top_p": 0.9,
    "do_sample": true,
    "max_new_tokens": 1024
  }
}
JSON

printf 'Validando Compose experimental...\n'
docker compose -f "$COMPOSE_FILE" config > "${OUTPUT_DIR}/compose-config.yml"
printf 'Construindo/verificando apenas a imagem experimental (sem pesos)...\n'
docker compose -f "$COMPOSE_FILE" build "$SERVICE"
printf 'Subindo apenas %s em %s...\n' "$SERVICE" "$BASE_URL"
docker compose -f "$COMPOSE_FILE" up -d --no-deps "$SERVICE"
wait_for_ready

curl --silent --show-error --fail "${BASE_URL}/v1/info" > "${OUTPUT_DIR}/info.json"
python3 - "${OUTPUT_DIR}/info.json" "$MODEL_ID" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if info.get("model") != sys.argv[2]:
    raise SystemExit("/v1/info não informa o modelo 0.6B efetivamente carregado")
PY

printf 'Executando aquecimento (não entra nas estatísticas)...\n'
generate_audio warmup no
DEVICE_VRAM_BEFORE="$(vram_used_mib)"
[[ "$DEVICE_VRAM_BEFORE" =~ ^[0-9]+([.][0-9]+)?$ ]] || die 'não foi possível obter VRAM global antes das medições.'

for index in 1 2 3 4 5; do
  label="$(printf 'run-%02d' "$index")"
  printf 'Executando %s/05 sequencialmente...\n' "$label"
  if [[ "$index" == "1" ]]; then
    generate_audio "$label" yes
  else
    generate_audio "$label" no
  fi
done

DEVICE_VRAM_AFTER="$(vram_used_mib)"
[[ "$DEVICE_VRAM_AFTER" =~ ^[0-9]+([.][0-9]+)?$ ]] || die 'não foi possível obter VRAM global após as medições.'
[[ -s "${OUTPUT_DIR}/nvidia-smi-dmon.log" ]] || die 'nvidia-smi dmon não produziu saída durante run-01.'

python3 "${SCRIPT_DIR}/benchmark_report.py" \
  --output-dir "$OUTPUT_DIR" \
  --device-vram-before-mib "$DEVICE_VRAM_BEFORE" \
  --device-vram-after-mib "$DEVICE_VRAM_AFTER"

printf '\nBenchmark concluído sem parar ou reiniciar aegis-tts (1.7B).\n'
printf 'Relatórios: %s/summary.json e %s/summary.md\n' "$OUTPUT_DIR" "$OUTPUT_DIR"
printf 'WAVs: %s/run-01.wav ... %s/run-05.wav\n' "$OUTPUT_DIR" "$OUTPUT_DIR"
