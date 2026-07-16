#!/usr/bin/env bash
# Download exactly the approved Qwen Base repository into the persistent
# project volume.  It never removes models, caches, images, or volumes.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DIR_HOST="${PROJECT_ROOT}/models/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DIR_CONTAINER="/models/Qwen3-TTS-12Hz-1.7B-Base"

die() {
  printf 'Erro: %s\n' "$*" >&2
  exit 1
}

confirm_download() {
  if [[ "${AEGIS_TTS_ASSUME_YES:-}" == "1" ]]; then
    printf 'Confirmação recebida por AEGIS_TTS_ASSUME_YES=1.\n'
    return 0
  fi
  if [[ ! -t 0 ]]; then
    die 'é necessária confirmação interativa; use AEGIS_TTS_ASSUME_YES=1 somente em automação consciente.'
  fi
  local answer
  read -r -p 'Baixar/retomar somente este modelo? [s/N] ' answer
  [[ "$answer" =~ ^([sS]|[sS][iI][mM])$ ]] || {
    printf 'Download cancelado; nada foi removido.\n'
    exit 0
  }
}

warn_if_low_space() {
  # The exact upstream snapshot size can change, so this is deliberately a
  # warning rather than a guessed hard limit. It prevents a surprising partial
  # download while preserving the user's decision and all existing data.
  local available_kib
  available_kib="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
  if [[ "$available_kib" =~ ^[0-9]+$ ]] && ((available_kib < 8 * 1024 * 1024)); then
    printf '\nAVISO: há menos de 8 GiB livres. O download pode não caber junto com a imagem e o cache.\n' >&2
    printf 'Nenhum arquivo será removido automaticamente; libere espaço conscientemente antes de continuar.\n\n' >&2
  fi
}

cd "$PROJECT_ROOT"
command -v docker >/dev/null 2>&1 || die 'Docker não encontrado.'
docker compose version >/dev/null 2>&1 || die 'Docker Compose não encontrado.'

printf 'Modelo autorizado: %s\n' "$MODEL_ID"
printf 'Diretório persistente no host: %s\n' "$MODEL_DIR_HOST"
printf 'Diretório no container: %s\n\n' "$MODEL_DIR_CONTAINER"
printf 'Espaço em disco antes:\n'
df -h "$PROJECT_ROOT"
warn_if_low_space
if [[ -d "$MODEL_DIR_HOST" ]]; then
  printf '\nTamanho atual do diretório do modelo (se parcial, será retomado): '
  du -sh "$MODEL_DIR_HOST" || true
fi

# Required before confirmation so the subsequent explicit model operation uses
# a known local image.  This build never downloads model weights.
if ! docker image inspect aegis-tts:local >/dev/null 2>&1; then
  printf '\nImagem aegis-tts:local ausente; construindo a imagem (sem pesos do modelo)...\n'
  docker compose build aegis-tts
else
  printf '\nImagem aegis-tts:local já existe; não será reconstruída.\n'
fi

printf '\nEspaço disponível imediatamente antes da confirmação:\n'
df -h "$PROJECT_ROOT"
warn_if_low_space

confirm_download
mkdir -p "$MODEL_DIR_HOST" "$PROJECT_ROOT/cache/home" "$PROJECT_ROOT/cache/xdg" "$PROJECT_ROOT/cache/numba" "$PROJECT_ROOT/samples"
# Bind mounts are host-owned. Prepare only this project's writable mounts;
# this never changes unrelated Docker data or user files.
chown -R "$(id -u):$(id -g)" "$PROJECT_ROOT/cache" "$PROJECT_ROOT/models" "$PROJECT_ROOT/samples" "$PROJECT_ROOT/voices"

printf '\nIniciando download explícito de %s...\n' "$MODEL_ID"
# Compose sets offline mode for normal service startup.  Override it only in
# this explicit downloader invocation; no other model ID is supplied.
docker compose run --rm --no-deps \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --entrypoint hf \
  aegis-tts download "$MODEL_ID" --repo-type model --local-dir "$MODEL_DIR_CONTAINER"

printf '\nDownload concluído. Tamanho final dos pesos/arquivos do repositório:\n'
du -sh "$MODEL_DIR_HOST"
printf '\nEspaço em disco depois:\n'
df -h "$PROJECT_ROOT"
