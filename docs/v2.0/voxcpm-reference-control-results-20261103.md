# VoxCPM2: controle com referência de treino 0586

O condicionamento acústico é viável na A2000, mas ainda não está aprovado
para o produto. No checkpoint800, tornou alguns indicadores de voz mais
estáveis e acrescentou cerca de100ms até a primeira fala. O ASR marcou duas
edições adicionais em87palavras; a identidade robótica não foi avaliada por
escuta. O próximo controle possível é repetir a comparação no checkpoint1400,
antes de escolher referência ou pesos finais.

## Execução e limites

O job095m terminou em03/11/2026,17:41UTC, com fonte do piloto
`adff50ab6e8d939075968eb326033ed8edd8bcc9de32e976c079fea15e87dc8d`.
Carregou o checkpoint800 salvo (`b9160041…501f33`), LoRA r8 BF16 sem fusão,
backbone BF16 e VAE FP32, com todos os pesos, buffers e KVs na GPU.
Usou KV1024, NFE10, CFG2, seed40733133, max_len400 e frontend numérico.
A referência0586 pertence ao treino; as três âncoras de validação serviram
somente à análise. Teste final6 e target50 continuam fechados.

Foram18 chamadas fixas de equivalência entre APIs,18 warmups e18 chamadas
medidas: três textos, dois modos e n3. O último par sem referência foi
bitidêntico em FLOAT/PCM nos três textos. A primeira rodada command diferiu;
as duas seguintes coincidiram. Todos os controles tiveram saída válida.
Não houve tolerância numérica nem repetição até coincidir.

A matriz A/B passou sem falhas, NaN, overshoot, truncamento indicado pelo
limite ou lacunas acima de10ms. Houve16.776 replays reais durante warmup.
Fonte, checkpoint, referência, manifests, configurações, base e VAE passaram
pelas verificações de integridade antes/depois. O treino retomou do passo2215
e confirmou nova atualização de pesos.

## Tempo no cliente nativo

São medianas de n3 por texto/modo; p95 não é estimado. A é sem referência;
B usa o cache fixo0586. Inclui prefill, staging das entradas, decodificação,
FIR48→24k e PCM. Não é HTTP nem inclui a reserva de reprodução320ms.

| Caso | Primeira fala disponível A/B | TTFA de reprodução A/B | RTF A/B | Lacunas A/B |
| --- | ---: | ---: | ---: | ---: |
| command | 240,9 /344,6ms | 280,9 /379,1ms | 0,758 /0,704 | 0 /0 |
| pressure | 234,7 /342,4ms | 274,7 /376,1ms | 0,693 /0,698 | 0 /0 |
| long | 244,6 /352,5ms | 284,6 /387,4ms | 0,682 /0,683 | 0 /0 |

O primeiro PCM ficou em aproximadamente235–247ms nos dois modos. B gerou
mais silêncio inicial, que a regra comum corretamente não conta como fala.
O cache acústico foi construído uma vez em3,194s, com65.536bytes FP32 na CPU;
o encoder VAE executou na GPU. Esse staging de entradas não descarrega pesos.
O prefill é refeito por pedido. O custo frio deve entrar na prontidão de um
futuro serviço; não foi somado artificialmente ao TTFA aquecido.

## Conteúdo e voz

Whisper-small local fixo, CPUint8/2threads, português, beam5, temperatura0,
sem VAD/prompt/contexto anterior: seis WAVs reais avaliados, um por texto/modo.
Verbatim A/B:12/87 contra14/87 edições (WER13,79%/16,09%); CER38/504 contra
37/504 (7,54%/7,34%). A referência falada/efetiva teve12/92 contra14/92.
Não se misturam estes denominadores com as sete frases do benchmark completo.

Os casos command e pressure tiveram a mesma contagem de erros A/B; as duas
edições extras ocorreram em long. Os alinhamentos mostram principalmente
grafias do nome próprio e números, além de inserções de artigo. Isso não
determina sozinho o que foi pronunciado. A regressão de WER supera2p.p. neste
subconjunto; clareza não está aprovada e não se descarta essa diferença como
erro do ASR sem revisão do áudio.

| Indício auxiliar, mediana entre três clipes | A | B | Âncoras reais |
| --- | ---: | ---: | ---: |
| Cosseno para âncoras | 0,98823 | 0,99005 | — |
| F0 | 208,4Hz | 207,3Hz | 200,1Hz |
| DP das três medianas F0 | 4,91Hz | 0,45Hz | 2,56Hz |
| Centroide | 913,2Hz | 747,8Hz | 897,6Hz |
| Energia8–12kHz | 2,494% | 1,608% | 2,962% |

O cosseno sobe em dois casos e cai no longo. A própria referência TRAIN0586
tem centroide673,5Hz e energia8–12kHz1,341%; a queda de B acompanha um
exemplar real mais escuro. Seu F0 mediano é196,2Hz. Essa referência não foi
incluída nas âncoras nem usada para recalibrar limiares.

O encoder pertence ao Qwen e não é um verificador calibrado deste domínio.
Textos, prosódia e gravações confundem cosseno e espectro; YIN pode errar
vozeamento/oitava. HF não mede diretamente metalização ou sibilância.
Não houve escuta nem veredito automático de identidade. F5 especializado e
Qwen processado permanecem controles nos mesmos três textos no relatório
auxiliar; seus resultados não aprovam a candidata por ordenação de cosseno.

## Artefatos e decisão

Raiz privada: `/var/tmp/aegis-v2-work/voxcpm2/`.

- `reference0586-step800-warmed-api-v2/`: execução, receitas, métricas e WAVs.
- `reference0586-control/results-analysis-v2/`: snapshot vinculado ao CP,
  ASR, alinhamentos, identity15, descritores da referência e recibo consolidado.
- `reference0586-control/failed-first-cold-0fcb/`: primeira tentativa preservada,
  que parou antes de B. Não é uma comparação com referência.

A primeira chamada ao CLI ASR desta análise teve um conflito de nome de
campo antes de transcrever áudio. A adaptação de metadados está registrada;
os seis áudios e textos não mudaram. Não se contabiliza esse erro de interface
como falha da síntese. As análises de identidade foram CPU, sem CUDA; não se
repetiu ASR após o resultado íntegro.

Manter o prompt como controle promissor, com clareza e identidade ainda
inconclusivas. Uma réplica em outro checkpoint pode testar a estabilidade dos
indícios; eventual seleção exige o conjunto completo de validação e medição
real pelo gateway320ms. O treino principal continua por validação.
