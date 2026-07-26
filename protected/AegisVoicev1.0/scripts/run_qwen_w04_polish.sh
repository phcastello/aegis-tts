#!/usr/bin/env bash
# Source-locked Q00 -> W04 polish.  It does not start Qwen, vLLM, or Docker.
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dsp="$root/private/environments/dsp/bin/python"

if [[ ! -x "$dsp" ]]; then
  "$root/scripts/create_dsp_environment.sh"
fi

exec "$dsp" "$root/scripts/aegis_robotic_voice.py" qwen-w04-polish "$@"
