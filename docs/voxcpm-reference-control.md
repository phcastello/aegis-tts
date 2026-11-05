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
cancellation. The paired A2000 control completed on 2026-11-05. It is an
experimental validation result; no production winner has been selected.

Both fresh processes loaded the same strict-origin step-6000 adapter, used NFC
text, seed 40733133, NFE 10, CFG 2, passthrough audio, PCM 24 kHz mono and startup 320 ms,
with three external warmups and three repetitions of each common validation case.
The only intervention was one fixed TRAIN acoustic reference, with no caption.
Each arm completed 21/21 benchmark requests and the WAV, PCM, SSE and cancellation
contracts. The no-reference arm reproduced all seven previously retained WAVs
and PCM files byte for byte. Final-test sentences remain unevaluated.

| Measurement | No reference | Fixed TRAIN reference |
| --- | ---: | ---: |
| Short-command first-speech p50, client ms | 701.64 | 768.31 |
| Short-reply first-speech p50, client ms | 678.50 | 789.49 |
| Long first-speech p50, client ms | 699.21 | 789.18 |
| Long synthesis RTF | 0.68542 | 0.69644 |
| RAW ASR word edits / 135 | 13 | 12 |
| Spoken ASR word edits / 154 | 37 | 37 |
| Auxiliary mean speaker cosine | 0.986918 | 0.989311 |

The three latency observations per case describe this control, not a robust p95.
The matched Qwen comparison with three external warmups is still pending.
Reference-cache preparation took 25.151 s before readiness; warm request timing
includes reference conditioning. All 21 reference-arm requests had positive
instrumented CUDA graph replay. Model, VAE and KV residency passed runtime guards;
this does not add an independent tensor inspection.

The auxiliary cosine gain was +0.002392; aggregate F0, spectral centroid and 4–8 kHz
energy moved closer to the reserved real anchors. These are imperfect indicators,
not an auditory identity verdict. The reference arm introduced unresolved ASR
flags for the name Aegis and HTTPS despite similar aggregate error counts.
No audio was heard by the agent, and the optional listening packet remains blind.
Pronunciation and identity therefore do not yet justify promotion. The preexisting
specialized F5 remains a separate comparator, not a generic replacement.

One terminal observer identity alert occurred after the reference launcher's clean
result and before worker reaping. The original alert is preserved alongside its
timing reconciliation; no resource stop or signal was issued. Both jobs exited 0,
their owned processes disappeared, GPU emptiness was confirmed, and the protected
Qwen container remained stopped within the authorized training window.
