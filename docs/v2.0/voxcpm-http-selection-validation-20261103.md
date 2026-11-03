# VoxCPM2: entrega HTTP dos passos1800,2800 e4000

O passo2800 é o candidato nominal desta validação para clareza automatizada:
teve15/135 edições ASR contra17/135 em1800 e18/135 em4000. Os três começam
a falar cerca de1,12s antes do Qwen pelo mesmo critério no cliente, sem gaps
nas63 requisições. As diferenças de velocidade entre eles cabem no jitter
observado. A identidade robótica e a superioridade ao F5 permanecem
inconclusivas sem escuta; nenhum modelo foi aprovado no teste final.

Foram medidos os mesmos sete textos de validação, três repetições por sistema
(`n21` de timing). O ASR e o passe de identidade usam somente rep0 (`n7` por
sistema), sem selecionar a melhor repetição. Os baselines Qwen/F5 foram
vinculados aos WAVs exatos por SHA. Todos os21 timings e sete ASRs de cada
sistema abaixo foram mantidos; não houve falhas excluídas dos denominadores.

| Sistema | TTFA p50 agregado (ms) | Faixa observada (ms) | RTF longo p50 | Gaps | Timing/saída |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen produção processado | 1827,6 | 1721,3–1948,3 | 0,6547 | 1 (22,3ms) | HTTP/PCM,21/21 |
| F5 especializado NFE16 | 2339,9 | 1698,1–2865,0 | 0,2549 | 0 | Arquivos por sentença,21/21 |
| Vox1800 | 711,5 | 686,1–1251,0 | 0,7294 | 0 | HTTP/PCM,21/21 |
| Vox2800 | 708,8 | 700,5–1107,0 | 0,7231 | 0 | HTTP/PCM,21/21 |
| Vox4000 | 705,1 | 686,6–1033,0 | 0,7222 | 0 | HTTP/PCM,21/21 |

O ganho mediano nominal de2800 contra Qwen é1118,8ms (61,2%). A mediana de
fim de síntese dos textos mistos é5485,1ms em2800 e4203,8ms no Qwen; começar
antes não significa terminar antes. O RTF longo permanece inferior a1 nos
três Vox. O F5 usa checkpoint especializado SHA`b3faf799…`, sem DSP extra;
entrega a sentença completa depois de cada ODE, sem HTTP ou streaming neural
dentro do bloco. Seu tempo inclui frontend, vocoder, conversão e fechar WAV.
Esse escopo distinto é preservado na comparação, sem streaming hipotético.

Por caso: **p50 [mínimo–máximo]ms, n3**. Não se estima p95 com três observações
por texto; os máximos e outliers continuam visíveis. A mediana agregada acima
mistura comprimentos e não substitui esta tabela.

| Caso | Qwen | F5 NFE16 | Vox1800 | Vox2800 | Vox4000 |
| --- | --- | --- | --- | --- | --- |
| v_command | 1844 [1775–1915] | 1878 [1863–1913] | 711 [710–734] | 712 [707–728] | 703 [702–736] |
| v_reply | 1911 [1752–1913] | 1713 [1698–1715] | 752 [749–753] | 710 [708–713] | 690 [687–1033] |
| v_pressure | 1813 [1721–1881] | 2583 [2340–2702] | 707 [706–776] | 712 [703–795] | 708 [705–1032] |
| v_acronyms | 1852 [1828–1948] | 2457 [2321–2699] | 721 [704–881] | 706 [702–936] | 705 [704–996] |
| v_question | 1804 [1786–1811] | 2353 [2319–2727] | 697 [686–1231] | 709 [704–778] | 707 [702–818] |
| v_sibilants | 1800 [1759–1841] | 2513 [1811–2548] | 718 [706–1251] | 713 [706–1107] | 707 [702–710] |
| v_long | 1828 [1820–1838] | 2708 [2327–2865] | 711 [711–711] | 707 [700–708] | 704 [703–718] |

Os Vox usam frontend `ptbr-numbers-v1`, reserva320ms, onset off, KV1024/NFE10,
CFG2/seed40733133, LoRA r8 BF16 sem fusão, AudioVAE FP32 e FIR41taps48→24.
Pesos, buffers e KVs permanecem na A2000; não há prompt de referência nem DSP
extra. Houve três warmups de uma frase fora do corpus na mesma thread,
somente antes de readiness. Rep0 é inédita depois de ready, sem prewarm por
caso. TTFA mede fala reproduzível no FIFO do cliente, incluindo a reserva e
silêncio inicial detectado; não mede o dispositivo físico. Primeiro PCM p50
foi571,5/568,8/566,6ms; a diferença até TTFA reflete o início silencioso.

Cold readiness1800/2800/4000:166,76/166,88/166,68s, com cache de compilação
existente naquela janela. Pico amostrado startup6084/6084/6082MiB e matriz
6086MiB nos três. Cada servidor foi encerrado;63/63 requests ficaram
`completed`. WAV mono24k completo, PCM real, DELETE após primeiro PCM,
cancelamento, próxima geração e health/ready/fila passaram nos três runs.

ASR fixo: faster-whisper-small pinado, CPU int8/duas threads, idioma pt,
beam5/temperatura0, sem VAD, trim, ganho ou prompt. O script
SHA`4ee9f7da…` e as configurações são idênticos aos recibos dos baselines.
WER/CER são comparações de transcrição; não confirmam pronúncia pelo ouvido.

| Sistema,7/7 ASRs completos | WER verbatim | CER verbatim | WER spoken | WER efetivo do próprio frontend |
| --- | --- | --- | --- | --- |
| Qwen | 21/135=15,56% | 55/755=7,28% | 39/154=25,32% | Indisponível no recibo original |
| F5 especializado NFE16 | 26/135=19,26% | 57/755=7,55% | 36/154=23,38% | 37/157=23,57% |
| Vox1800 | 17/135=12,59% | 44/755=5,83% | 35/154=22,73% | 20/142=14,08% |
| Vox2800 | 15/135=11,11% | 43/755=5,70% | 34/154=22,08% | 18/142=12,68% |
| Vox4000 | 18/135=13,33% | 43/755=5,70% | 37/154=24,03% | 21/142=14,79% |

`verbatim` compara os mesmos textos solicitados; `spoken` usa a lista comum
pré-fixada; `effective` compara a redação realmente entregue ao modelo. O
recibo histórico F5 chama essa última comparação de`frontend`; seu texto é
diferente, e os denominadores157/142 não permitem tratá-la como ranking
homogêneo entre arquiteturas. Os CER spoken/efetivos e diferenças por caso
ficam no resumo privado, sem recalcular ou reescrever as transcrições.

Entre1800/2800/4000, as edições verbatim por caso são1/1/1 no comando,
0/0/0 na resposta,6/5/6 em pressão,3/2/3 em siglas,0/0/0 na pergunta,
1/1/1 em sibilantes e6/6/7 no longo. Nomes próprios, algarismos versus
palavras e URL geram divergências; há também `envie` reconhecido como
`envia` e `seção` como `sessão`. Em1800, CPF foi reconhecido como CTF;
2800/4000 recuperam CPF, mas continuam divergências na URL. São alertas para
verificação auditiva, não uma aprovação de clareza. A menor loss de4000
não produziu o menor WER desta entrega compilada.

O passe38 contém três âncoras reais reservadas, Qwen7, F5 especializado7 e
os21 WAVs HTTP novos. Foi usado o helper imutável SHA`c9cc80bd…`, mesma
política RMS ativo−23dBFS e encoder local do Qwen. Não houve nova síntese ou
encoder baixado, nem âncora usada como prompt.

| Sistema | Cosseno mediano auxiliar | F0 mediano (Hz) | Centroide mediano (Hz) | Energia8–12kHz |
| --- | ---: | ---: | ---: | ---: |
| Âncoras reais,3 textos diferentes | — | 200,06 | 897,64 | 2,962% |
| Qwen | 0,981865 | 220,31 | 539,09 | 0,803% |
| F5 especializado NFE16 | 0,989979 | 207,29 | 1152,12 | 4,294% |
| Vox1800 | 0,988533 | 206,68 | 1047,30 | 3,113% |
| Vox2800 | 0,986955 | 188,75 | 952,58 | 2,798% |
| Vox4000 | 0,985638 | 184,96 | 932,42 | 2,996% |

Os indicadores discordam:1800 tem maior cosseno entre os três Vox;4000 tem
centroide/banda alta mais próximos de alguns agregados das âncoras, e2800
tem menor WER. O encoder pode favorecer Qwen, os textos das âncoras diferem,
YIN pode errar oitavas/trechos surdos e bandas não são um teste de metalicidade.
Nenhum áudio foi ouvido e não há limiar automático de identidade.

ASR21 consumiu183,89s internos do auditor; identidade38,21,42s/RSS987,93MiB,
CPU2/nice19/CUDA não inicializada. O leitor local terminou após executar
uma única rodada, com reconciliação de PID/saídas. Fontes, resultados e pesos
foram conferidos; testes finais seis/50 continuam fechados.

Artefatos privados novos em
`/var/tmp/aegis-v2-work/voxcpm-gateway/selection-analysis-1800-2800-4000/`:
snapshots de CP/WAVs, jobs ASR com aliases por passo/SHA, identidade38,
`execution-receipt.json`, `report-summary.private.json`, diferenças ASR e
`report-receipt.json` SHA`6d7891f30d580d6ad3832c9fef6f66fb6a558ae06c1f4aaa277782e7f249f881`.
Os rótulos HTTP iguais foram preservados em`upstream_candidate`; somente
os jobs de análise têm nomes distintos. Não se agregaram checkpoints pelo
nome comum do perfil.

Após os três probes, o operador removeu somente202.784.768 bytes de cache
descartável, com recibo. Isso não altera WAVs/hashes/timings já concluídos;
readiness após recompilar o cache ainda não foi medido. O bloqueio de espaço
do FastPitch096b não indica inviabilidade do modelo; o operador retomou096c.
O passo2800 pode seguir como candidato prático de **validação** por contratos,
espera e clareza automatizada. Confirmar identidade robótica e escolha final
continua exigindo a evidência reservada e a escuta pertinente.
