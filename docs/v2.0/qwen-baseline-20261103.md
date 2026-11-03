# Qwen: baseline local, 03/11/2026

A nova medição aquecida terminou em **21/21 respostas válidas**, sete frases
inéditas × três repetições, sem clipping PCM16. O gateway real inclui o perfil
robótico e streaming de produção, na RTX A2000 12 GB. Não houve escuta nesta
medição: fala detectada digitalmente não comprova clareza ou identidade.

| Frase / tamanho | Primeiro PCM, p50 s | Fala disponível, p50 s | TTFA FIFO, p50 s | TTFA FIFO mínimo–máximo s | EOF p50 s | RTF p50 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| comando / curta | 1,764 | 1,765 | 1,844 | 1,775–1,915 | 2,268 | 0,796 |
| resposta / curta | 1,811 | 1,811 | 1,911 | 1,752–1,913 | 2,423 | 0,891 |
| pressão/hora / média | 1,733 | 1,733 | 1,813 | 1,721–1,881 | 4,758 | 0,634 |
| siglas / média | 1,772 | 1,772 | 1,852 | 1,828–1,948 | 5,659 | 0,686 |
| pergunta / média | 1,711 | 1,711 | 1,804 | 1,786–1,811 | 4,204 | 0,702 |
| sibilantes / média | 1,720 | 1,720 | 1,800 | 1,759–1,841 | 4,034 | 0,650 |
| estabilidade / longa | 1,728 | 1,728 | 1,828 | 1,820–1,838 | 17,167 | 0,655 |

P95 indisponível com n=3 por frase. Uma resposta da pergunta teve uma lacuna FIFO
estimada de **22,252 ms**; as outras vinte não tiveram lacuna >10 ms. VRAM total
máxima amostrada: **10.674 MiB**, a cada 500 ms; não é pico do allocator e pode
omitir eventos breves. A duração do áudio varia entre repetições; não foi
selecionada a geração mais rápida nem excluído outlier.

A disponibilidade de fala é a chegada dos samples necessários para confirmar
duas janelas de 20 ms com RMS ≥ −45 dBFS. TTFA FIFO simula reprodução imediata
respeitando chegada e duração dos blocos; exclui latência do dispositivo.
O silêncio inicial detectado foi de 80–100 ms na primeira repetição. Métrica
completa e limitações em [protocolo](protocol-20261103.md).

Produção observada pelos headers: fonte `1749f5fb29320dced0a16add618989e36954ee7b`,
perfil `AegisVoicev1.0-Q00-W04-P07-M02`, DSP `stable`, runtime
`stable-codec22x22-onset-conservative-v1-startup320ms`. Saída PCM s16le mono,
24 kHz, HTTP chunked, `POST /v1/audio/speech`, `stream:true`. PIDs CUDA observados
155890/156571, GPU exclusiva desse backend. O harness não fez warm-up descartado.
A rodada aconteceu em 06:07:34–06:09:35 UTC; partida fria será registrada à parte.

Evidência privada compacta:
`/var/tmp/aegis-v2-work/bench/qwen-warm/`, com `manifest.json`, `results.jsonl`,
`summary.json`, `gpu.jsonl`, sete WAVs da repetição zero e hashes de todas as
respostas. O manifesto fixa corpus/harness/ambiente/payload. Áudios e textos
privados de referência não foram incluídos no Git.

Antes de qualquer síntese do teste final, o limiar foi selado em
`/var/tmp/aegis-v2-work/decision-policy.json`, SHA-256
`1d0592faea4d4bcfdab81e0e3785c64a7abc9a8fe9ae3aa88c42386a228e1137`.
O ganho material exigido é **400 ms de TTFA FIFO por frase curta/média**, com
continuidade no trecho longo. A fórmula usa o maior de 250 ms, 20% da mediana
Qwen e duas vezes `1,4826 × MAD`, arredondando o maior limiar por frase para cima
em passos de 50 ms. Os limiares sem arredondamento ficaram em 360,075–382,179 ms.
Com n=3, MAD tem baixa precisão; vantagem próxima ao ruído exige ampliar as
repetições pareadas. Regressão de WER acima de 2 pontos percentuais após revisar
divergências, erros críticos persistentes ou perda do caráter robótico reprovam.
A candidata selecionada será registrada separadamente, com evidência da
validação, antes de abrir o teste. Não há candidata aprovada neste baseline.

## Auditor ASR fixo, CPU

Os sete WAVs da repetição zero foram transcritos em CPU int8, duas threads,
um worker e nice 19, sem CUDA, prompt, VAD, ganho ou corte. Modelo local
`Systran/faster-whisper-small`, revisão
`536b0662742c02347bc0e980a01041f333bce120`; os quatro arquivos foram revalidados
contra o recibo fixado antes da carga. `faster-whisper==1.1.1`,
`ctranslate2==4.6.0`, idioma `pt`, beam 5, temperatura zero,
`condition_on_previous_text=False`. WAV→16 kHz float32 por `resample_poly`,
Kaiser beta 10. Não houve download nem instalação nesta rodada.

A normalização de WER é a histórica: NFD, remover acentos, casefold, separar
pontuação/espaços, preservar letras/dígitos. CER usa os mesmos tokens unidos
por um espaço, contando esses espaços. Nenhuma expansão é aplicada ao ASR.
Os textos esperados crus e os escritos por extenso foram fixados separadamente,
antes de inspecionar as transcrições; as duas comparações permanecem disponíveis.

| Referência textual | Edições / palavras | WER auxiliar | CER auxiliar |
| --- | ---: | ---: | ---: |
| Request verbatim | 21 / 135 | 15,56% | 7,28% |
| Forma de fala pretendida | 39 / 154 | 25,32% | 13,01% |

Esses percentuais **não são notas de pronúncia**: o ASR frequentemente escreve
números com algarismos e siglas sem soletração, aumentando a distância contra
formas por extenso. `v_reply` e `v_question` tiveram igualdade normalizada contra
o request cru. Houve divergências no nome, siglas, unidade, flexão de verbos e
`seção/sessão`, preservadas para revisão. O ASR reconheceu `Égide` no comando
e no texto longo; o frontend protegido já substitui Aegis dessa forma, conforme
auditoria histórica. Não se atribui automaticamente esse caso ao modelo acústico.
Não houve julgamento auditivo de inteligibilidade, timbre ou identidade.

Carga ASR: 1,194 s; rodada completa: 96,992 s. São custos do auditor CPU,
fora das métricas TTS. Recibos, hipóteses, segments, hashes dos WAVs e da entrada
ASR: `/var/tmp/aegis-v2-work/asr/qwen-warm.json`; agregado em
`qwen-warm-summary.json`. Comparar finalistas somente na mesma coorte de sete
frases e na mesma política textual. Reconstruções de vocoder ficam em outra
coorte, com as transcrições privadas originais.

Repetição, sem retranscrever WAV/texto iguais já presentes no recibo:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  nice -n 19 /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  benchmarks/v2/asr_score.py \
  --qwen-dir /var/tmp/aegis-v2-work/bench/qwen-warm \
  --output /var/tmp/aegis-v2-work/asr/qwen-warm.json
```
