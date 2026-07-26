#!/usr/bin/env bash
# Focused Q00 -> W04 sibilance comparison. It never starts Qwen, vLLM, or Docker.
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dsp="$root/private/environments/dsp/bin/python"

if [[ ! -x "$dsp" ]]; then
  "$root/scripts/create_dsp_environment.sh"
fi

exec "$dsp" "$root/scripts/aegis_robotic_voice.py" qwen-w04-polish "$@"
