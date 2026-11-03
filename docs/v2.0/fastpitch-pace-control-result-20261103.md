# FastPitch8000: resultado do controle pace1.0 versus0.8

O pace0.8 alongou corretamente o áudio, mas não demonstrou uma correção geral
de clareza. O WER verbatim passou de42 para41 edições em135 palavras, enquanto
o CER piorou de91 para105 em755 caracteres. Os resultados por frase são
mistos. Isso não encerra a linha FastPitch nem altera pace, pesos ou produto.

Foi usado o checkpoint8000 exato, SHA
`ae9834fb562562a405febfcaa325f57711779aeadda8ebfa933e12ce02c922c8`,
no helper independente SHA`67bee551…`, após revisão root/peer. São sete frases
de validação, n1 por pace,14 condições pré-fixadas, seed40733133 reiniciado,
ordem alternada, mesmo refinement do7000 e mesmos dois modelos num backend.
Todo o controle rodou em CPU FP32, duas threads/nice19, sem CUDA ou treino.
Esses tempos e scores não são uma medição de latência da A2000/GPU BF16.

| Comparação | Pace1.0 | Pace0.8 |
| --- | --- | --- |
| TTS / ASR completos | 7/7 / 7/7 | 7/7 / 7/7 |
| WER verbatim | 42/135=31,11% | 41/135=30,37% |
| CER verbatim | 91/755=12,05% | 105/755=13,91% |
| WER spoken | 60/154=38,96% | 55/154=35,71% |
| WER do texto efetivo | 59/157=37,58% | 58/157=36,94% |
| CER do texto efetivo | 145/841=17,24% | 160/841=19,02% |

Nenhuma falha foi excluída:14/14 TTS e14/14 ASRs concluíram. Na saída FLOAT,
NaNs, overshoots e amostras no limite intrínseco do BigVGAN foram zero; na
conversão PCM não houve clipping, ganho, trim ou limitação adicional.
O ASR fixo CPU int8 é o mesmo scorer SHA`4ee9f7da…`, sem reescrever textos,
VAD ou prompt. Seus erros continuam auxiliares, sem avaliação auditiva.

| Caso | Duração1.0→0.8 (s) | Edições verbatim1.0→0.8 |
| --- | --- | --- |
| v_command | 2,869→3,605 | 3→1 |
| v_reply | 2,528→3,168 | 2→3 |
| v_pressure | 7,008→8,736 | 8→8 |
| v_acronyms | 6,709→8,416 | 8→12 |
| v_question | 6,133→7,669 | 2→2 |
| v_sibilants | 6,069→7,563 | 1→2 |
| v_long | 23,968→29,963 | 18→13 |

O ganho concentra-se no comando e no longo; resposta, siglas e sibilantes
pioram no ASR. Há divergências em palavras comuns além de nome/números/siglas,
portanto o ganho líquido de uma edição não estabelece um gate de idioma.
N1 por caso também não permite estimar variabilidade nem atribuir esses
resultados diretamente a fonemas ou à percepção do usuário.

O fator observado de duração foi1,2460–1,2565 (mediana1,2504). As somas de
duração previstas antes de pace foram idênticas nos sete pares; os frames
coincidiram com o arredondamento FP32 do regulador. A hipótese inicial usava
o ratio de validação7500=0,7704. O valor observado8000=0,791676 dividido por0.8
é aproximadamente0,9896, mas esse agregado teacher-forced não é uma duração
ideal conhecida para os novos textos. O arredondamento por grafema/token
também impede prometer um fator exato por frase.

Somadas as14 gerações, o custo interno CPU foi669,23s; o longo custou151,33s
em1.0 e198,94s em0.8. O ASR consumiu187,76s internos, sequencialmente depois
do TTS e do ASR público8000 anterior. O pico RSS do TTS foi2,368GiB, abaixo
do limite5GiB; não houve abortamento por recurso. Os WAVs/JSONs ocupavam cerca
de6MB na conclusão TTS, abaixo de50MB. O processo de controle terminou.

Recibos e14 PCM pareados privados em
`/var/tmp/aegis-v2-work/fastpitch/pace-control-step8000-eval-v2/`:
`control-summary.private.json` SHA`c29c399b…`, `control-receipt.json`
SHA`6538537c…` e `temporary-checkpoint-cleanup.json` SHA`ffe72744…`.
Os hashes de pesos/fontes foram conferidos antes/depois. O hardlink
temporário próprio de549.622.954 bytes foi removido; o retainer mantém o
mesmo8000 sob sua política, sem uma retenção adicional deste experimento.
Por existir outro vínculo ao mesmo inode, essa remoção não liberou blocos de
pesos. Metadados, receita, métricas e os poucos megabytes de comparação ficam.

Nenhum serviço ou treino foi sinalizado, nenhum pace de produto foi alterado,
e os conjuntos final6/alvo50 continuam fechados. A próxima decisão de treino
de idioma deve usar a validação completa; este controle apenas mostra que
expandir durações globalmente não resolveu o problema observado em8000.
