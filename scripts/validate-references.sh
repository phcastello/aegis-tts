#!/usr/bin/env bash
# Validate local reference assets in place.  No conversion, deletion, or
# transcription is performed; this is a read-only preflight report.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VOICE_IDS=(giseli-10s giseli-34s)
TABLE_FILE="$(mktemp "$PROJECT_ROOT/cache/.reference-validation.XXXXXX")"
ERROR_COUNT=0

cleanup() {
  rm -f -- "$TABLE_FILE"
}
trap cleanup EXIT

has_text() {
  [[ -f "$1" ]] && grep -q '[^[:space:]]' -- "$1"
}

below() {
  awk -v value="$1" -v limit="$2" 'BEGIN { exit !(value < limit) }'
}

above_or_equal() {
  awk -v value="$1" -v limit="$2" 'BEGIN { exit !(value >= limit) }'
}

command -v python3 >/dev/null 2>&1 || {
  printf 'Erro: python3 é necessário para validar WAV PCM.\n' >&2
  exit 1
}
command -v sha256sum >/dev/null 2>&1 || {
  printf 'Erro: sha256sum é necessário.\n' >&2
  exit 1
}

printf 'ID\tarquivo\tcodec\tsample rate\tcanais\tbits\tduração\ttamanho\thash áudio\thash transcrição\ttranscrição presente\tstatus\n' >"$TABLE_FILE"

for voice_id in "${VOICE_IDS[@]}"; do
  audio="$PROJECT_ROOT/voices/$voice_id/reference.wav"
  transcript="$PROJECT_ROOT/voices/$voice_id/reference.txt"
  status=()
  codec='-'
  sample_rate='-'
  channels='-'
  bits='-'
  duration='-'
  size='-'
  audio_hash='-'
  transcript_hash='-'
  transcript_present='não'

  if [[ ! -f "$audio" ]]; then
    status+=("ERROR: WAV ausente")
    ((ERROR_COUNT+=1))
  else
    audio_hash="$(sha256sum -- "$audio" | awk '{print $1}')"
    if info="$(python3 "$SCRIPT_DIR/audio-info.py" --tsv "$audio" 2>/dev/null)"; then
      :
    else
      # The helper still produces a structured row on an invalid WAV.
      :
    fi
    IFS=$'\t' read -r codec sample_rate channels bits duration _size_bytes size peak rms silence clipping readable inspect_error <<<"$info"
    if [[ "$readable" != "True" ]]; then
      status+=("ERROR: WAV inválido${inspect_error:+ ($inspect_error)}")
      ((ERROR_COUNT+=1))
    else
      if [[ "$codec" != "PCM" ]]; then
        status+=("ERROR: codec não-PCM")
        ((ERROR_COUNT+=1))
      fi
      if [[ "$channels" != "1" ]]; then
        status+=("ERROR: mono esperado")
        ((ERROR_COUNT+=1))
      fi
      if below "$duration" 3 || above_or_equal "$duration" 60.000001; then
        status+=("ERROR: duração fora de 3–60 s")
        ((ERROR_COUNT+=1))
      fi
      if [[ "$sample_rate" != "32000" ]]; then
        status+=("WARN: sample rate diferente de 32000 Hz")
      fi
      if above_or_equal "$clipping" 0.001000; then
        status+=("WARN: clipping severo ≥0,1%")
      fi
      if above_or_equal "$silence" 0.800000; then
        status+=("WARN: silêncio excessivo ≥80%")
      fi
    fi
  fi

  if has_text "$transcript"; then
    transcript_present='sim'
    transcript_hash="$(sha256sum -- "$transcript" | awk '{print $1}')"
  elif [[ -f "$transcript" ]]; then
    status+=("ERROR: transcrição vazia")
    ((ERROR_COUNT+=1))
  else
    status+=("ERROR: transcrição ausente")
    ((ERROR_COUNT+=1))
  fi

  if ((${#status[@]} == 0)); then
    status=(OK)
  fi
  status_text=''
  for issue in "${status[@]}"; do
    [[ -z "$status_text" ]] || status_text+='; '
    status_text+="$issue"
  done
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$voice_id" \
    "$(basename -- "$audio")" \
    "$codec" "$sample_rate" "$channels" "$bits" "$duration" "$size" \
    "$audio_hash" "$transcript_hash" "$transcript_present" "$status_text" >>"$TABLE_FILE"
done

if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$TABLE_FILE"
else
  cat "$TABLE_FILE"
fi

printf '\nCritérios: duração entre 3 e 60 s; WAV PCM mono; 32.000 Hz é aceito sem conversão.\n'
printf 'Pico/RMS/silêncio/clipping são calculados sem alterar o arquivo.\n'

if ((ERROR_COUNT > 0)); then
  printf '\nValidação falhou com %d erro(s). Nenhum arquivo foi alterado.\n' "$ERROR_COUNT" >&2
  exit 1
fi
printf '\nValidação concluída sem erros. Avisos não provocam conversão automática.\n'
