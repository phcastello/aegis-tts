#!/usr/bin/env bash
# User-invoked only. This starts a lab-owned vLLM container per cold profile,
# generates Qwen sources with Aninha, then applies strict L00 and W04 effects.
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dsp="$root/private/environments/dsp/bin/python"

if [[ ! -x "$dsp" ]]; then
  "$root/scripts/create_dsp_environment.sh"
fi

exec "$dsp" "$root/scripts/aegis_robotic_voice.py" qwen-cold-run "$@"
