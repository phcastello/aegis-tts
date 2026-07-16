#!/usr/bin/env bash
# End-to-end smoke test for an already downloaded model and two imported
# private references. It keeps the service running and writes four new
# timestamped WAVs below this project's samples/smoke directory.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/models/Qwen3-TTS-12Hz-1.7B-Base"
SMOKE_DIR="$PROJECT_ROOT/samples/smoke"
SMOKE_TIMEOUT_SECONDS="${AEGIS_TTS_SMOKE_TIMEOUT_SECONDS:-900}"
REQUEST_TIMEOUT_SECONDS="${AEGIS_TTS_SMOKE_REQUEST_TIMEOUT_SECONDS:-600}"

die() {
  printf 'Erro: %s\n' "$*" >&2
  exit 1
}

# Read a simple KEY=value value without sourcing .env as shell code. Values
# are never echoed; this is used for the optional API key and published port.
dotenv_value() {
  local key="$1"
  local dotenv="$PROJECT_ROOT/.env"
  [[ -f "$dotenv" ]] || return 0
  awk -v wanted="$key" '
    /^[[:space:]]*#/ { next }
    index($0, "=") > 0 {
      candidate = substr($0, 1, index($0, "=") - 1)
      if (candidate == wanted) {
        print substr($0, index($0, "=") + 1)
        exit
      }
    }
  ' "$dotenv" | tr -d '\r'
}

published_port="${AEGIS_TTS_PORT:-}"
[[ -n "$published_port" ]] || published_port="$(dotenv_value AEGIS_TTS_PORT)"
published_port="${published_port:-8001}"
[[ "$published_port" =~ ^[0-9]+$ ]] && ((published_port >= 1 && published_port <= 65535)) || die 'AEGIS_TTS_PORT inválida.'
BASE_URL="${AEGIS_TTS_SMOKE_BASE_URL:-http://127.0.0.1:${published_port}}"
api_key="${AEGIS_TTS_API_KEY:-}"
[[ -n "$api_key" ]] || api_key="$(dotenv_value AEGIS_TTS_API_KEY)"
AUTH_ARGS=()
if [[ -n "$api_key" ]]; then
  AUTH_ARGS=(-H "X-API-Key: $api_key")
fi

command -v docker >/dev/null 2>&1 || die 'Docker não encontrado.'
docker compose version >/dev/null 2>&1 || die 'Docker Compose não encontrado.'
command -v curl >/dev/null 2>&1 || die 'curl não encontrado.'
command -v jq >/dev/null 2>&1 || die 'jq é necessário para montar o payload JSON.'
command -v python3 >/dev/null 2>&1 || die 'python3 é necessário para inspecionar os WAVs retornados.'

[[ -f "$MODEL_DIR/config.json" ]] || die "modelo ausente em $MODEL_DIR; execute ./scripts/download-model.sh primeiro."
[[ -f "$PROJECT_ROOT/voices/giseli-10s/reference.wav" ]] || die 'referência giseli-10s ausente; execute import-reference.sh primeiro.'
[[ -f "$PROJECT_ROOT/voices/giseli-34s/reference.wav" ]] || die 'referência giseli-34s ausente; execute import-reference.sh primeiro.'

cd "$PROJECT_ROOT"
printf 'Validando referências antes do smoke test...\n'
"$SCRIPT_DIR/validate-references.sh"

mkdir -p "$SMOKE_DIR" "$PROJECT_ROOT/cache"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TMP_FILES=()
cleanup() {
  local file
  for file in "${TMP_FILES[@]}"; do
    [[ -n "$file" ]] && rm -f -- "$file"
  done
}
trap cleanup EXIT

printf '\nIniciando/atualizando o serviço...\n'
docker compose up -d --build aegis-tts

printf 'Aguardando %s/health/ready (limite: %ss)...\n' "$BASE_URL" "$SMOKE_TIMEOUT_SECONDS"
deadline=$((SECONDS + SMOKE_TIMEOUT_SECONDS))
status_code=''
while ((SECONDS < deadline)); do
  status_code="$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' --connect-timeout 3 "$BASE_URL/health/ready" || true)"
  if [[ "$status_code" == "200" ]]; then
    break
  fi
  sleep 5
done
[[ "$status_code" == "200" ]] || {
  docker compose logs --tail=100 aegis-tts || true
  die "serviço não ficou pronto dentro de ${SMOKE_TIMEOUT_SECONDS}s (último HTTP: ${status_code:-sem resposta})."
}

printf '\nHealth atual:\n'
curl --silent --show-error "$BASE_URL/health" | jq .

synthesize() {
  local voice="$1"
  local text="$2"
  local output="$3"
  local payload headers response
  payload="$(mktemp "$PROJECT_ROOT/cache/.smoke-payload.XXXXXX.json")"
  headers="$(mktemp "$PROJECT_ROOT/cache/.smoke-headers.XXXXXX.txt")"
  response="$(mktemp "$PROJECT_ROOT/cache/.smoke-response.XXXXXX.wav")"
  TMP_FILES+=("$payload" "$headers" "$response")

  jq -n \
    --arg text "$text" \
    --arg voice "$voice" \
    --arg language 'pt-BR' \
    --argjson seed 42 \
    '{text: $text, voice: $voice, language: $language, seed: $seed,
      generation_parameters: {}}' >"$payload"

  printf '\nSintetizando %s...\n' "$voice"
  if ! curl --silent --show-error --fail-with-body \
      --connect-timeout 10 --max-time "$REQUEST_TIMEOUT_SECONDS" \
      -D "$headers" \
      -H 'Content-Type: application/json' \
      "${AUTH_ARGS[@]}" \
      --data-binary "@$payload" \
      "$BASE_URL/v1/synthesize" \
      --output "$response"; then
    printf '\nResposta de erro para %s (se houver):\n' "$voice" >&2
    sed -n '1,80p' "$response" >&2 || true
    return 1
  fi

  mv -- "$response" "$output"
  printf 'WAV salvo: %s\n' "$output"
  printf 'Métricas da resposta:\n'
  grep -i '^X-Aegis-TTS-' "$headers" | tr -d '\r' || true
  python3 "$SCRIPT_DIR/audio-info.py" --human "$output"
}

CASUAL_TEXT='Olá, Pedro. O serviço de voz da Aegis está funcionando corretamente. Pelo que eu verifiquei, está tudo normal por enquanto.'
TECHNICAL_TEXT='A Aegis integra PostgreSQL, Qdrant, Docker, Entity Framework e serviços OpenAI.'
OUTPUT_10S_CASUAL="$SMOKE_DIR/${TIMESTAMP}-giseli-10s-casual.wav"
OUTPUT_34S_CASUAL="$SMOKE_DIR/${TIMESTAMP}-giseli-34s-casual.wav"
OUTPUT_10S_TECHNICAL="$SMOKE_DIR/${TIMESTAMP}-giseli-10s-tecnico.wav"
OUTPUT_34S_TECHNICAL="$SMOKE_DIR/${TIMESTAMP}-giseli-34s-tecnico.wav"

# The service queue is intentionally serial: these calls are sequential and
# exercise each reference with identical text, seed and native model defaults.
synthesize giseli-10s "$CASUAL_TEXT" "$OUTPUT_10S_CASUAL"
synthesize giseli-34s "$CASUAL_TEXT" "$OUTPUT_34S_CASUAL"
synthesize giseli-10s "$TECHNICAL_TEXT" "$OUTPUT_10S_TECHNICAL"
synthesize giseli-34s "$TECHNICAL_TEXT" "$OUTPUT_34S_TECHNICAL"

printf '\nVRAM global após o teste:\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
else
  printf 'nvidia-smi não encontrado no host.\n'
fi
printf '\nLogs recentes do serviço:\n'
docker compose logs --tail=100 aegis-tts || true
printf '\nSmoke test concluído. Os quatro WAVs foram mantidos em %s\n' "$SMOKE_DIR"
