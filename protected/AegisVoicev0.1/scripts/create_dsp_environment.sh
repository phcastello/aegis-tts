#!/usr/bin/env bash
# Create only the private CPU DSP environment required by the Qwen cold lab.
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dsp="$root/private/environments/dsp"

if [[ -x "$dsp/bin/python" ]]; then
  echo "DSP environment already exists: $dsp"
  exit 0
fi

python3 -m venv "$dsp"
"$dsp/bin/python" -m pip install --upgrade pip
rg -v '^pyworld' "$root/requirements-dsp.txt" | "$dsp/bin/python" -m pip install -r /dev/stdin -e "$root"

# PyWorld has no CPython 3.12 wheel. Headers are unpacked privately, with no
# global package or production/vLLM environment modification.
build="$root/private/build/python-dev"
mkdir -p "$build"
(cd "$build" && apt-get download libpython3.12-dev python3.12-dev && for pkg in ./*.deb; do dpkg-deb -x "$pkg" extracted; done)
CFLAGS="-I$build/extracted/usr/include/python3.12 -I$build/extracted/usr/include" CXXFLAGS="-I$build/extracted/usr/include/python3.12 -I$build/extracted/usr/include" "$dsp/bin/python" -m pip install "setuptools==80.9.0" "pyworld==0.3.5"
echo "Created private DSP environment: $dsp"
