# AegisVoice v1.0 — proteção de reprodução

Este diretório congela a seleção auditiva `Q00 → WORLD W04` de 2026-07-25:
frase `sibilants`, aperiodicidade vozeada `0.52` (P07) e mastering `M02`
(de-esser leve). O WAV entregue é `audio/q00-w04-mastered-sibilants.wav`.

## Conteúdo

- `config/`: configuração final e catálogos exatos usados na seleção.
- `audio/`: fonte Q00, WORLD W04 raw P07 e W04 mastered M02 do mesmo texto.
- `reference/`: referência Aninha e transcrição exata.
- `manifests/`: evidência da geração Qwen e dos sweeps P07/mastering.
- `code/` e `scripts/`: cópia exata do laboratório usado.
- `dependencies/`: declaração e freeze do ambiente DSP.
- `provenance/`: commit, imagem Docker por digest e hashes.

## Seleção final

- Fonte: Q00 `sibilants`.
- WORLD W04: F0 sem modulação, quantização 200 cents, transição zero,
  `formant_scale: 1.10`, `aperiodicity_voiced_scale: 0.52`.
- Mastering M02: passa-altas 60 Hz, de-esser centrado em 6.5 kHz (banda 2.2
  kHz, limiar −30 dBFS, redução máxima 1.5 dB, ataque 2 ms, release 45 ms) e
  limitador em −1 dB.

Verifique a integridade a partir deste diretório com:

```bash
sha256sum -c provenance/SHA256SUMS
```
