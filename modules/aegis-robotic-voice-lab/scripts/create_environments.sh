#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
chat="$root/private/environments/chatterbox"
python3 -m venv "$chat"
"$chat/bin/python" -m pip install --upgrade pip
# Chatterbox 0.1.7 requires the official PyTorch 2.6 pair. Validate CUDA at
# preflight; this remains contained in the private environment.
"$chat/bin/python" -m pip install "torch==2.6.0" "torchaudio==2.6.0"
"$chat/bin/python" -m pip install "git+https://github.com/resemble-ai/chatterbox.git@5de7a54aa4e5e2baadb0182dde554908b48b85c2" huggingface_hub safetensors soundfile
"$root/scripts/create_dsp_environment.sh"
echo "Created private Chatterbox and DSP environments. No global or vLLM environment was changed."
