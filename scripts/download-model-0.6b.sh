#!/usr/bin/env bash
# Download only the isolated 0.6B Base checkpoint. This script never removes
# or rewrites the production 1.7B directory, Docker objects, or global caches.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.0.6b.yml"
MODEL_ID="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_DIR_HOST="${PROJECT_ROOT}/models/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_DIR_CONTAINER="/models/Qwen3-TTS-12Hz-0.6B-Base"

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

validate_checkpoint() {
  [[ -f "${MODEL_DIR_HOST}/config.json" ]] || die "config.json ausente em ${MODEL_DIR_HOST}"
  python3 - "$MODEL_DIR_HOST" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
if (model_dir / "model.safetensors").is_file():
    print("Pesos principais validados: model.safetensors")
    raise SystemExit(0)
index_path = model_dir / "model.safetensors.index.json"
if not index_path.is_file():
    raise SystemExit("Pesos principais ausentes: model.safetensors ou model.safetensors.index.json")
try:
    filenames = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
except (OSError, ValueError, KeyError, AttributeError) as exc:
    raise SystemExit(f"Índice de pesos inválido: {exc}")
missing = sorted(name for name in filenames if not (model_dir / name).is_file())
if not filenames or missing:
    raise SystemExit("Pesos fragmentados ausentes: " + ", ".join(missing or ["weight_map vazio"]))
print(f"Pesos principais validados: índice com {len(filenames)} arquivo(s)")
PY
}

cd "$PROJECT_ROOT"
command -v docker >/dev/null 2>&1 || die 'Docker não encontrado.'
docker compose version >/dev/null 2>&1 || die 'Docker Compose não encontrado.'
[[ -f "$COMPOSE_FILE" ]] || die "Compose experimental ausente: ${COMPOSE_FILE}"

printf 'Modelo autorizado: %s\n' "$MODEL_ID"
printf 'Destino persistente no host: %s\n' "$MODEL_DIR_HOST"
printf 'Destino no container: %s\n\n' "$MODEL_DIR_CONTAINER"
printf 'Espaço em disco disponível antes do download:\n'
df -h "$PROJECT_ROOT"
if [[ -d "$MODEL_DIR_HOST" ]]; then
  printf '\nDiretório 0.6B existente (um download parcial será retomado): '
  du -sh "$MODEL_DIR_HOST" || true
fi

confirm_download

printf '\nConstruindo/verificando somente a imagem experimental aegis-tts:0.6b (sem pesos)...\n'
docker compose -f "$COMPOSE_FILE" build aegis-tts-0.6b

# Prepare only this model's own directory and shared writable cache locations.
# In particular, do not recursively chown ./models, which contains the 1.7B.
mkdir -p "$MODEL_DIR_HOST" "$PROJECT_ROOT/cache/home" "$PROJECT_ROOT/cache/xdg" \
  "$PROJECT_ROOT/cache/numba"
chown -R "$(id -u):$(id -g)" "$MODEL_DIR_HOST" \
  "$PROJECT_ROOT/cache/home" "$PROJECT_ROOT/cache/xdg" "$PROJECT_ROOT/cache/numba"

printf '\nIniciando download explícito de apenas %s...\n' "$MODEL_ID"
# The normal Compose service is offline. These two overrides apply only to this
# invocation of the Hugging Face CLI and this one exact repository.
docker compose -f "$COMPOSE_FILE" run --rm --no-deps \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --entrypoint hf \
  aegis-tts-0.6b download "$MODEL_ID" --repo-type model --local-dir "$MODEL_DIR_CONTAINER"

printf '\nValidando checkpoint 0.6B baixado...\n'
validate_checkpoint
printf '\nTamanho final do diretório 0.6B: '
du -sh "$MODEL_DIR_HOST"
printf '\nEspaço em disco disponível após o download:\n'
df -h "$PROJECT_ROOT"
printf '\nConcluído sem remover, substituir ou limpar o modelo 1.7B.\n'
