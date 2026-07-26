# AegisVoice v0.1 — proteção de reprodução

Este diretório preserva a seleção auditiva `Q00 → W04` de 2026-07-25.
O exemplo é `neutral-short`; o áudio entregue ao ouvinte é
`audio/q00-w04-mastered-neutral-short.wav`.

## Conteúdo

- `config/`: seleção canônica, matriz Qwen, template vLLM e deploy Q00 real.
- `audio/`: fonte Q00, W04 raw e W04 mastered do mesmo texto.
- `reference/`: Aninha e sua transcrição exata.
- `manifests/`: resultados completos do lote Q00–Q04, incluindo falhas.
- `code/` e `scripts/`: cópia exata do código de laboratório usado.
- `dependencies/`: declaração e freeze do ambiente DSP.
- `provenance/`: commit Git, imagem Docker por digest e hashes.

## Reprodução de áudio já ouvido

Para preservar a identidade que foi aprovada, use o WAV fonte Q00 já salvo
como entrada para WORLD W04. Uma inferência Qwen nova com o mesmo seed mantém
a direção, mas não é uma garantia bit a bit.

Verifique a integridade com `sha256sum -c provenance/SHA256SUMS` a partir deste
diretório.
