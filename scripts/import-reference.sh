#!/usr/bin/env bash
# Import a local WAV and its exact supplied transcript without ever modifying
# the audio.  The two known profile IDs prevent path traversal or arbitrary
# writes below voices/.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

die() {
  printf 'Erro: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Uso:
  ./scripts/import-reference.sh <giseli-10s|giseli-34s> /caminho/reference.wav /caminho/reference.txt

O texto deve ser a transcrição exata já fornecida para o WAV. Este script não
transcreve, não converte e não normaliza áudio.
EOF
}

has_text() {
  grep -q '[^[:space:]]' -- "$1"
}

confirm_overwrite() {
  if [[ "${AEGIS_TTS_ASSUME_YES:-}" == "1" ]]; then
    printf 'Sobrescrita confirmada por AEGIS_TTS_ASSUME_YES=1.\n'
    return 0
  fi
  if [[ ! -t 0 ]]; then
    die 'as referências de destino já existem; é necessária confirmação interativa para sobrescrever.'
  fi
  local answer
  read -r -p 'Os arquivos existentes serão substituídos. Continuar? [s/N] ' answer
  [[ "$answer" =~ ^([sS]|[sS][iI][mM])$ ]] || {
    printf 'Importação cancelada; arquivos existentes foram preservados.\n'
    exit 0
  }
}

[[ $# -eq 3 ]] || { usage >&2; exit 2; }
VOICE_ID="$1"
SOURCE_WAV="$2"
SOURCE_TEXT="$3"

[[ "$VOICE_ID" =~ ^[a-z0-9_-]+$ ]] || die 'ID inválido; use apenas a-z, 0-9, hífen e sublinhado.'
case "$VOICE_ID" in
  giseli-10s|giseli-34s) ;;
  *) die 'ID não configurado. Os únicos IDs permitidos são giseli-10s e giseli-34s.' ;;
esac

command -v python3 >/dev/null 2>&1 || die 'python3 é necessário para validar WAV PCM localmente.'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum é necessário.'
[[ -f "$SOURCE_WAV" && -r "$SOURCE_WAV" ]] || die "WAV de origem ausente ou ilegível: $SOURCE_WAV"
[[ -f "$SOURCE_TEXT" && -r "$SOURCE_TEXT" ]] || die "transcrição de origem ausente ou ilegível: $SOURCE_TEXT"
has_text "$SOURCE_TEXT" || die 'a transcrição está vazia; forneça a transcrição exata do áudio.'

SOURCE_WAV_REAL="$(realpath -e -- "$SOURCE_WAV")"
SOURCE_TEXT_REAL="$(realpath -e -- "$SOURCE_TEXT")"
TARGET_DIR="$PROJECT_ROOT/voices/$VOICE_ID"
TARGET_WAV="$TARGET_DIR/reference.wav"
TARGET_TEXT="$TARGET_DIR/reference.txt"
TARGET_WAV_REAL="$(realpath -m -- "$TARGET_WAV")"
TARGET_TEXT_REAL="$(realpath -m -- "$TARGET_TEXT")"

[[ "$SOURCE_WAV_REAL" != "$TARGET_WAV_REAL" ]] || die 'a origem WAV já é o destino; não há nada para importar.'
[[ "$SOURCE_TEXT_REAL" != "$TARGET_TEXT_REAL" ]] || die 'a origem de transcrição já é o destino; não há nada para importar.'

printf 'Informações técnicas do WAV de origem (somente leitura):\n'
python3 "$SCRIPT_DIR/audio-info.py" --human "$SOURCE_WAV_REAL" || die 'o arquivo não é um WAV PCM legível; ele não foi copiado.'
printf '\nSHA-256 do WAV: %s\n' "$(sha256sum -- "$SOURCE_WAV_REAL" | awk '{print $1}')"
printf 'SHA-256 da transcrição: %s\n' "$(sha256sum -- "$SOURCE_TEXT_REAL" | awk '{print $1}')"

if [[ -e "$TARGET_WAV" || -e "$TARGET_TEXT" ]]; then
  printf '\nDestino existente: %s\n' "$TARGET_DIR"
  confirm_overwrite
fi

mkdir -p "$TARGET_DIR"
TMP_WAV="$(mktemp "$TARGET_DIR/.reference.wav.XXXXXX")"
TMP_TEXT="$(mktemp "$TARGET_DIR/.reference.txt.XXXXXX")"
cleanup() {
  rm -f -- "${TMP_WAV:-}" "${TMP_TEXT:-}"
}
trap cleanup EXIT

# cp preserves the supplied master bytes; no decoder, resampler, or metadata
# rewriting is involved.
cp --preserve=mode,timestamps -- "$SOURCE_WAV_REAL" "$TMP_WAV"
cp --preserve=mode,timestamps -- "$SOURCE_TEXT_REAL" "$TMP_TEXT"

SOURCE_WAV_HASH="$(sha256sum -- "$SOURCE_WAV_REAL" | awk '{print $1}')"
SOURCE_TEXT_HASH="$(sha256sum -- "$SOURCE_TEXT_REAL" | awk '{print $1}')"
[[ "$SOURCE_WAV_HASH" == "$(sha256sum -- "$TMP_WAV" | awk '{print $1}')" ]] || die 'falha de integridade ao copiar o WAV.'
[[ "$SOURCE_TEXT_HASH" == "$(sha256sum -- "$TMP_TEXT" | awk '{print $1}')" ]] || die 'falha de integridade ao copiar a transcrição.'

mv -f -- "$TMP_WAV" "$TARGET_WAV"
mv -f -- "$TMP_TEXT" "$TARGET_TEXT"
trap - EXIT

printf '\nReferência importada sem alterar o áudio:\n'
printf '  WAV: %s\n' "$TARGET_WAV"
printf '  texto: %s\n' "$TARGET_TEXT"
printf '  SHA-256 WAV: %s\n' "$SOURCE_WAV_HASH"
printf '  SHA-256 texto: %s\n' "$SOURCE_TEXT_HASH"
printf '\nExecute ./scripts/validate-references.sh antes de iniciar o serviço.\n'
