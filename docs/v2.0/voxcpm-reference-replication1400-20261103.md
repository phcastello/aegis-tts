# Referência0586: réplica no checkpoint1400

A réplica confirma que a referência fixa altera a voz de forma consistente
em alguns descritores, mantendo a inferência viável na A2000. A identidade
robótica continua sem aprovação auditiva. O prompt não foi incorporado ao
gateway; o treino principal segue acompanhado pela validação.

Usou a mesma receita do [controle800](voxcpm-reference-control-results-20261103.md),
mudando somente o checkpoint salvo para1400:
`d79948253574f4bd6d6dc8ba7f8247389b852cf6d35d62c5df21b3a3a0ba6e8a`.
Piloto `adff50ab…dc8d`, KV1024, NFE10, CFG2, seed40733133, max_len400,
frontend numérico, LoRA BF16 sem fusão e VAE FP32. Todos os pesos e KVs
permaneceram na GPU. Referência0586 TRAIN; âncoras VAL apenas para avaliação.
Teste final6 e target50 permanecem fechados.

O job095o terminou com sucesso em03/11/2026,18:41UTC. As18 equivalências
fixas,18 warmups e18 medições A/B concluíram; último par de cada texto
bitidêntico entre APIs sem referência. Fonte, entradas, pesos-base/VAE e
checkpoint passaram pelas verificações antes/depois. Foram17.532 replays
reais no warmup. A matriz terminou sem falhas, gaps>10ms, NaN, overshoot ou
limite de geração atingido. O treino retomou do estado completo2618 e
confirmou avanço com novos deltas de pesos.

## Tempo nativo, n3 por caso e modo

A=sem referência; B=TRAIN0586. Medianas; p95 não estimado. Inclui staging,
prefill, geração, VAE, FIR48→24 e PCM. Não é o gateway/startup320ms.

| Caso | Primeira fala disponível A/B | TTFA de reprodução A/B | RTF A/B |
| --- | ---: | ---: | ---: |
| command | 371,6 /351,2ms | 471,6 /405,8ms | 0,799 /0,707 |
| pressure | 340,4 /341,6ms | 375,0 /396,3ms | 0,690 /0,698 |
| long | 240,1 /609,7ms | 340,1 /673,7ms | 0,686 /0,695 |

Houve variação local importante: primeiro PCM command/A242–453ms e
long/B241–533ms. A causa não foi atribuída. B também gerou mais silêncio
inicial em alguns casos. Não se transforma essa matriz curta em uma
afirmação de latência do produto; o gateway exigirá nova medição completa.
O cache fixo da referência levou3,191s para construir,65.536bytes FP32CPU,
pico alocado5629,5MiB durante esse estágio. Não é offload de pesos.

## Conteúdo e descritores de voz

ASR fixo idêntico ao controle800: seis WAVs, um por texto/modo, CPUint8,
2threads/nice19, português/beam5/temperatura0, sem VAD/prompt/contexto.
Verbatim A/B12/87 versus13/87 (WER13,79%/14,94%); CER38/504 versus40/504
(7,54%/7,94%). Falado/efetivo12/92 versus13/92. A diferença de1,15p.p.
neste subconjunto não substitui o gate sobre a validação completa.

Command e pressure tiveram1/1 e6/6 edições A/B; long5/6. O ASR não transcreveu
o nome Aegis no início da pressão/B. Isso exige escuta do WAV preservado;
não se declara omissão real nem erro do reconhecedor por suposição.
Há também grafias de números e nomes próprios que confundem WER/CER.

| Indício auxiliar, mediana entre três clipes | A | B | Âncoras reais |
| --- | ---: | ---: | ---: |
| Cosseno para âncoras | 0,98996 | 0,99024 | — |
| F0 | 207,9Hz | 212,1Hz | 200,1Hz |
| DP das três medianas F0 | 12,00Hz | 2,03Hz | 2,56Hz |
| Centroide | 957,2Hz | 731,3Hz | 897,6Hz |
| Energia8–12kHz | 2,509% | 1,476% | 2,962% |

O cosseno subiu nos três pares, mas a mudança da mediana é pequena. B volta
a aproximar o espectro do exemplar TRAIN0586, que tem centroide673,5Hz e
energia8–12kHz1,341%. Essa referência permanece fora das âncoras. A menor
dispersão de F0 não comprova timbre robótico ou prosódia adequada. O encoder
Qwen é enviesado/não calibrado; YIN e energia espectral têm limitações de
vozeamento, texto e gravação. Nenhum áudio foi julgado auditivamente pela IA.

Identidade15 levou13,865s em CPU, pico956,9MiB, CUDA não inicializada.
Não houve repetição de ASR nem alteração dos helpers, do trainer ou do runtime.

## Escuta opcional e recuperação

Foi preparado um teste de validação com três pares cegos do CP1400 e uma
âncora real, para decidir se vale manter o condicionamento por referência.
A chave fica separada. Player atenua por RMS ativo dentro de cada par;
isso não é LUFS nem equivalência perceptiva comprovada. Os bytes dos WAVs
não mudaram: seis gerados mono24k/PCM16 e uma âncora original48k/PCM24.

ZIP local4.542.225bytes, SHA
`5c983c11cd8fd9dc4eab13b9c73bcc71a7fbf8a3e2b7e41940c668728c707d72`:
`/var/tmp/aegis-v2-work/listening/escuta-validacao-1400.zip`.
Feedback é opcional e não interrompe o treino. Não houve aprovação inferida.
O pacote e os áudios privados permanecem locais.

Artefatos sob `/var/tmp/aegis-v2-work/voxcpm2/`:

- `reference0586-step1400-warmed-api-v2/`: receita e medição nativa completa.
- `reference0586-control/results-analysis-step1400-v2/`: snapshot dos seis
  pares, ASR, identidade15, summaries, recibo consolidado e recibo do ZIP.
- `reference0586-control/prepare-results-analysis-step1400-v2.py`: preparação
  vinculada ao checkpoint, hashes, mesmos textos e estado de retomada.
- `window-watch-095o-095p-20261103/`: supervisão e prova de retomada2618→2624.

Decisão: manter a referência como opção experimental. Os dois controles
mostram alterações reproduzíveis dos descritores; pronúncia e caráter robótico
ainda exigem revisão. A escolha final conserva o teste reservado fechado até
fixar pesos e política por validação.
