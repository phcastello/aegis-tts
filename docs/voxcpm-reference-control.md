# Experimental VoxCPM2 acoustic reference control

The experimental delivery launcher accepts the operator arguments
`--reference-wav` and `--reference-sha256` together. The path must be absolute
and the hash must be an exact lowercase SHA256. Both are bound into the frozen
plan, runtime configuration and artifact manifest before the server starts.
Reference selection is an operator setting; speech requests do not select files.
Keep private recordings and their experiment manifests outside Git.

Omitting both arguments preserves the previous no-reference profile and serialized
runtime configuration. Enabling the control adds `-AcousticReference` to the
experimental profile. The protected Qwen profile remains a separate profile.
The control uses acoustic reference mode without a caption or continuation text.
It does not merge generated speech back into the reference cache.

This bounded experiment requires exactly 64 reference patches. The pinned VAE
encodes the reference on the GPU once, before readiness and warmup, and returns
finite FP32 input features of shape `[64, 4, 64]` on the CPU. Each request transfers
that small input prefix; model weights, VAE and KV caches remain on the GPU.
References producing another shape fail before readiness. Silence trimming is
disabled. The 1024-slot capacity guard reserves `text_tokens + 468` slots with
the reference, compared with `text_tokens + 402` without it, and rejects overflow
without truncation.

The native reference iterator yields waveform, token and feature tuples. The
adapter forwards only the waveform and closes the native iterator on cancellation
or error. Reference preparation time belongs to startup; per-request prefix
processing belongs to end-to-end synthesis and first-speech measurements.

For a comparison, use the same selected checkpoint, frontend, seed, inference
steps, CFG, output format, startup buffering and client onset rule in both fresh
processes. Record all attempts, including errors. The no-reference arm must use
the same runtime version as the reference arm. A local gain remains provisional
until pronunciation, streaming contracts and the common Qwen comparison pass.

CPU tests cover configuration, profile and serialized-default compatibility,
artifact and plan binding, cache shape and reuse, capacity boundaries and iterator
cancellation. Real A2000 reference loading, quality and timing qualification are
still pending. No voice-identity or latency improvement is claimed by this change.
