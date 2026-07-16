# Qwen3-TTS / vLLM-Omni isolated benchmark

This experiment never modifies the production Compose file. It serves the
existing 1.7B model on `127.0.0.1:8004`, mounts model and voice data read-only,
and uses `cache/vllm-omni` exclusively.

Pinned primary runtime:

- vLLM 0.22.0
- vLLM-Omni 0.22.0, commit `963ba1ab7a02b2c1f687285fcd2072cd57cd0b34`
- image `vllm/vllm-omni@sha256:8b0cc5438eb27b34cdfd22011b735da6a94835a09e9c56ddf9e8cb300d679919`
- CUDA 13.0.2, PyTorch 2.11.0

The image is configured with NVIDIA CUDA forward compatibility because the
host is on driver 535. This must be validated by the compatibility script; it
is not assumed from image metadata alone.

The Compose file uses an isolated 8 GiB `/dev/shm`, not `ipc: host`. The
official connector is process-shared-memory based, and the bounded shm mount
is the narrower choice.

Run:

```bash
scripts/vllm-compatibility-check.sh
scripts/vllm-benchmark.sh
scripts/vllm-compare.sh
```

`vllm-benchmark.sh` refuses to proceed if another GPU container besides
`aegis-tts` is running, or if safe free-disk headroom is unavailable. It stops
only `aegis-tts`, installs a restoration trap, and records the temporary
availability window.
