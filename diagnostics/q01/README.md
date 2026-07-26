# Q01 source-sampling diagnosis

These overlays inherit the protected Q00 deploy configuration and alter only
Stage 0 sampling.  They retain Aninha, seed `40733133` as their server default,
the Q00 connector settings, and do not involve WORLD, P07, M02, or production.

The runner recreates only `qwen-vllm-clean` on port 8004.  It saves raw PCM and
`metrics.tsv`; it does not run ASR or modify the protected profile.

Run the short, decisive sequence first:

```bash
AEGIS_Q01_SEEDS=40733133 diagnostics/q01/run_matrix.sh \
  S00-q00-original S01-repetition-105 S02-topk-5 S03-topk-10
```

Run the complete five-seed corpus when choosing a production candidate:

```bash
diagnostics/q01/run_matrix.sh
```

Set `AEGIS_Q01_HTTP_STREAM=true` to repeat a matrix through HTTP streaming.
Both modes emit raw `s16le`, 24 kHz, mono PCM; HTTP streaming is no longer the
variable under test.
