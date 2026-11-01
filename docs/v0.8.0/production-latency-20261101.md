# Produção TTS e preparação da nova voz — 01/11/2026

## Escopo e estado preservado

Trabalho iniciado no HEAD atualizado `6645539`, branch
`perf/v0.8.0-host-first-audio`, sem alterações locais iniciais. Nenhuma mudança
na main ou no repositório principal da Aegis. Produção confirmada antes de operar:
`aegis-tts-gateway:latency-ba1beb9`, fonte
`ba1beb9eff78eedf1df1b71ceb94b8539bccad43`, imagem
`sha256:78046393d0c3c712ad0ea1a278d8f909f482de477a89fa888907196198f31aaa`.

Permanecem Qwen3-TTS-12Hz-0.6B-Base, referência/idioma/sampling, perfil
`AegisVoicev1.0-Q00-W04-P07-M02`, WORLD W04/P07/M02, velocidade, 24 kHz mono
s16le, endpoints, WAV, SSE, prioridade, cancelamento e limites. O worker e a
serialização da referência já existiam; não são anunciados como ganhos novos.

Host: RTX A2000 12 GB (11.514 MiB visíveis), 12 CPUs lógicas Xeon E5-2470 v2,
23,47 GiB RAM. Antes, GPU em repouso ocupava 10.674 MiB. Não foi iniciado outro
modelo, treino, tráfego periódico ou cache de respostas. A troca é só do serviço
Compose `aegis-tts`; cache/pesos/mounts protegidos permanecem no mesmo local.

## Gargalo e matriz pequena

O chunk inicial anterior tinha 80 ms quase silenciosos a ~320–350 ms de relógio.
O lote seguinte tinha 2 s e trazia a primeira fala, perto de 1,4 s; só então a
janela WORLD estável conseguia trabalhar (~220–290 ms). O prefixo silencioso
continuava sendo reproduzido depois da chegada do PCM. Um pacote silencioso não
é voz disponível. O tempo de kernel/codec interno por ID não é exposto pelo runtime.

A fonte/schema instalados vLLM/Omni 0.22.0 foram inspecionados: o campo
`initial_codec_chunk_frames` existe e é transferido ao estágio assíncrono.
As capturas confirmam 16→1.280 ms e 22→1.760 ms na primeira chegada. O lote
seguinte vem do YAML do conector, não de campos arbitrários no payload.
`non_streaming_mode` top-level continua ignorado pela API; Base efetivo continua
false, e Q00 é imposto pelo YAML. Essa discrepância não foi usada como atalho.

| Configuração | Evidência e decisão |
| --- | --- |
| 1×25 + WORLD estável | Baseline real; raw incremental não contém fala útil no primeiro pacote. |
| 16×25 + janela inicial quieta de 900 ms | 12 gerações independentes; alguns textos antecipam, outros fazem fallback. Replay pós-trim chega a 793 ms de underrun. Rejeitada para produção. |
| 22×25 + WORLD estável | 12 gerações independentes; mediana do primeiro DSP processado ~1.337 ms em replay, mas pós-trim 10/12 têm gaps, até 450 ms. Não selecionada. |
| 22×22 + WORLD estável + corte conservador | Ensaio implantado sem reserva: primeiro adapter ~1,3–1,4 s; gaps até 297 ms nos textos adicionais. Não deixado como configuração final sem corrigir continuidade. |
| 22×22 + corte conservador + reserva de 320 ms | Variante final validada nas medições abaixo. A reserva tem custo explícito na chegada HTTP; não altera PCM nem velocidade. |

Há 36 capturas brutas novas (seis textos, duas repetições, três tamanhos iniciais)
e 48 replays. Todas as observações, inclusive as lentas, foram preservadas.
As capturas raw compartilham o backend operacional e não medem espera da fila
nativa; não se atribui automaticamente uma observação lenta ao codec.
Replays DSP usam exatamente os mesmos samples e agenda de chegada. Gerações
independentes têm variação acústica; igualdade bit a bit não é critério obrigatório
para o chunking. Não houve busca extensa nem alteração de modelo/referência.

## Mudanças e qualidade verificável

`OutputOnsetTrimmer` examina quadros de 10 ms. Só considera quieto um quadro com
pico ≤24 PCM16 e RMS ≤8 PCM16; abre imediatamente no primeiro quadro incerto,
preserva-o e 80 ms anteriores. Exige prefixo quieto ≥120 ms; probe limitado a 2 s.
Silêncio total/ambiguidade são preservados. O corte não é de duração fixa.
Todo o sufixo permanece bit a bit igual; não há fade, ganho, resampling, corte de
pausas internas ou caudas. Uma instância por unidade sintetizada preserva isolamento
por request_id e reduz a abertura redundante em unidades sucessivas. A cauda
anterior e o contrato de junção de 80–350 ms continuam preservados.

A cadência de 22 frames é derivada em `/tmp/aegis-runtime-deploy.yaml` por
`runtime_config.py`; somente `codec_chunk_frames` muda frente ao YAML protegido.
Nenhum arquivo protegido é reescrito. O override inicial 22 segue em cada pedido
interno. Todas as demais opções, contexto 72 frames, graph shapes e limites ficam
iguais. Binários e fontes Omni permanecem iguais à imagem anterior.

A reserva inicial de streaming é de 320 ms, limitada/configurável entre 0 e 500 ms,
com padrão de código 0. O produtor continua trabalhando na fila existente; o bloco
retirado segue contabilizado no limite de 4 s. Não há segunda fila ilimitada.
WAV não streaming não espera essa reserva. Cancelamento e falha durante a espera
descartam o bloco retido. Se a síntese já terminou, o estado histórico completed
é preservado, mas o cancelamento impede a entrega ainda não iniciada. Eventos
aditivos `stream_startup_buffering/released/cancelled` tornam essa espera visível.

A reserva foi escolhida sobre o pior gap observado de 297 ms, com 23 ms de margem;
é evidência desta coorte, não garantia para todo texto/carga/reprodutor. Fronteiras
WORLD menores não foram forçadas dentro de palavras para conseguir um número menor.
Não houve capacidade de escuta: identidade, primeiras consoantes, palavras, números,
prosódia e timbre ainda precisam de A/B do proprietário. A candidata é reversível,
com pendência de homologação auditiva, e não uma substituição por nova voz.

## Medição e resultados finais

Comparação no endpoint implantado, corpus principal, **9 chamadas antes e 9 depois**,
todas com fila inicialmente vazia e espera efetiva por segmento <0,3 ms. Todas
as primeiras observações foram mantidas. Valores em ms (média / mediana / máximo):

| Marco | Antes `ba1beb9` | Final `1749f5f` |
| --- | ---: | ---: |
| Headers HTTP | 3,7 / 3,0 / 4,9 | 3,5 / 3,0 / 4,7 |
| Primeiro PCM HTTP | 1635,2 / 1635,3 / 1666,2 | 1667,5 / 1652,8 / 1726,2 |
| Primeiro bloco contendo voz digital | 1635,3 / 1635,3 / 1666,8 | 1667,8 / 1653,1 / 1726,2 |
| Offset inicial no áudio | 485,9 / 588,6 / 592,9 | 95,4 / 94,6 / 106,2 |
| Primeira fala em reprodução estimada | 2121,1 / 2197,4 / 2254,9 | 1763,0 / 1755,9 / 1827,8 |
| EOF | 3726,1 / 2429,6 / 6647,9 | 3745,2 / 2588,0 / 6325,9 |

A mediana de primeira fala estimada caiu **441,4 ms (20,1%)**; a média caiu 358,1 ms. O primeiro PCM HTTP não melhorou: a reserva de continuidade consome parte do ganho de geração. Esses deltas são do endpoint, não soma de microganhos.

Nas seis chamadas adicionais comparáveis (duas repetições de três textos), a mediana
de primeira fala foi **1.888,9 → 1.797,6 ms**. Antes havia gaps em 2/6 (até
252 ms); depois, 0/6. Quatro chamadas de dois textos novos reservados, não usados
para escolher a reserva, mediram mediana **1.768,3 ms**, sem gaps. O corpus
principal passou de 0/9 para 0/9 gaps. No conjunto final: **19/19 sem underrun**
no simulador, inclusive sem gap positivo entre blocos; não é garantia universal.


A melhora varia por texto; comparação de medianas (ms):

| Texto | Primeira fala antes → depois | EOF antes → depois |
| --- | ---: | ---: |
| horário | 2241.5 → 1811.7 | 2099.6 → 2430.4 |
| matemática | 2197.4 → 1718.0 | 2429.6 → 2588.0 |
| longo | 1911.4 → 1747.4 | 6647.9 → 6203.2 |
| negativas | 1820.5 → 1834.7 | 3342.6 → 3037.9 |
| números/datas | 1888.9 → 1797.6 | 3602.3 → 3548.4 |
| consoantes | 1960.9 → 1723.9 | 3263.4 → 2889.4 |

Negativas ficaram cerca de 14 ms mais lentas nesta coorte; EOF de horário e
matemática aumentou, enquanto longo e adicionais diminuíram. A duração do áudio
varia entre gerações independentes, além do prefixo retirado; não se atribui toda
diferença de EOF ao código e não se afirma ganho uniforme.

Nos 19 primeiros segmentos, mediana de primeiro bruto/primeiro bruto contendo
voz: **1.085,0 / 1.085,1 ms**; chegada suficiente para DSP: **1.089,2 ms**, com
1.760 ms de áudio acumulados. Primeiro DSP CPU mediano: **248,7 ms**; fim DSP
**1.344,9 ms**; primeiro PCM processado **1.345,0 ms**, saída do adapter antes da
reserva **1.347,2 ms**. A reserva configurada em 320 ms esperou de **319,6 a
402,5 ms** (média 327,9 ms), pois event loop/DSP ainda têm escalonamento e GIL.
Ela está integralmente incluída nos números HTTP. O corte retirou **100–510 ms**
na abertura desses primeiros segmentos; não altera pausas internas/finais.

Os 15 ensaios sem reserva permanecem nos dados, com cinco pedidos iniciados com
fila ocupada e esperas de até ~5.964 ms no primeiro segmento. Houve chamadas
externas durante essa janela. A comparação final usa gate de fila somente por
GET, com espera fora do relógio do POST e registrada; ainda foi conferida a espera
real em todos os segmentos (máximo 0,294 ms). Não foram descartados lentos para
produzir a comparação. O gate não sintetiza nem mantém GPU aquecida.

Todos os 49 PCM HTTP preservados (antes, intermediário e final) têm comprimento
válido e **zero clipping**; pico máximo 22.365 PCM16. Contagem de samples por
adapter e junções explica a duração entregue. Gerações variaram em duração e
waveform, portanto estes checks não provam pronúncia nem ausência perceptiva de
omissões. Pares A/B identificados de seis textos ficam em
`/var/tmp/aegis-tts-latency-20261101/review/AB-index.json` e CSV; amostras DSP com
mesmo bruto ficam em `raw22/*-stable{,-onset}.wav`. Nada disso está no Git.

Resultados individuais sanitizados, incluindo os lentos, transport chunks,
eventos e replays: [host-benchmark-20261101.json](host-benchmark-20261101.json).
Proveniência/configuração numérica efetiva:
[host-runtime-20261101.json](host-runtime-20261101.json).

Tempos vêm de `perf_counter`/`monotonic` locais. HTTP é medido no processo cliente;
estágios DSP são medidos dentro do adapter, correlacionados pelo ID. Não foram
subtraídos relógios de máquinas/processos para inventar parcelas. `first_output`
do adapter e `audio_started` indicam geração/enqueue antes da reserva; a primeira
chegada HTTP e os eventos released incluem o custo dela.

O estimador de reprodução respeita a chegada real de cada sample e a fila FIFO.
`primeiro PCM + offset` sozinho é inválido se houver underrun. O limiar digital
`abs(PCM16)>100` é um proxy documentado, não escuta acústica ou medição Android.
Gaps menores que 10 ms afetam o relógio simulado, mas não entram no contador de
underruns. A meta de um segundo não é declarada atingida. A meta global de quatro
segundos também depende dos componentes externos, que não foram implantados aqui.

## Implantação, checks e rollback

Fonte final de runtime: `1749f5fb29320dced0a16add618989e36954ee7b`.
Imagem final: `aegis-tts-gateway:latency-1749f5f`.
Imagem/manifesto local:
`sha256:8aef74c67fe5238ce1e2b8781b9da3aedc73784014dd38bc1c1553f741fe93a0`.
Variante: `stable-codec22x22-onset-conservative-v1-startup320ms`.

Build usa `gateway/Dockerfile.runtime-overlay`, sobre os binários retidos
`6174bb8`. Commits Conventional Commits foram enviados à branch de trabalho;
dataset e preparação de treino têm commits separados. A imagem contém o gateway
da fonte indicada; commits posteriores de relatório não mudam esse runtime.
A recriação usa `docker compose up -d --no-build --no-deps aegis-tts`, mesma
identidade/projeto, portas, mounts, volumes/cache e assets. Não há `down -v`.

Profiling detalhado é opt-in por env ou arquivo operacional efêmero
`/tmp/aegis-tts-profile-latency`; o arquivo permite ligar/desligar a medição sem
outra recarga da GPU. O modo cotidiano é env 0 e arquivo ausente. Eventos não
contêm texto, referência ou credenciais.

Suíte final: **177 passed, 1 skipped** (teste GPU opt-in específico não acionado;
o smoke real usa a GPU operacional). Integridade do perfil, léxico, corpus de
pronúncia, golden regression e Compose passaram. Golden regression valida hashes,
não audição. Smoke no endpoint ativo passou em health, readiness, métricas/fila,
OpenAI WAV, Aegis WAV, PCM streaming, SSE, cancelamento ativo/enfileirado e
`reject_if_busy`. Os dois endpoints passaram também nos testes de integração de
PCM/WAV, cancelamento entre blocos e falha durante reserva.

A comparação do inspect e dos arquivos confirmou mesmos mounts (incluindo flags
read-only), portas, projeto, comando, GPU, shm/tmpfs e limites; gateway instalado
coincide com a fonte. Versões e hashes Omni são idênticos. A primeira recriação
experimental chegou a readiness em ~207,5 s; a final em **143,7 s**, sem reinício
para fabricar cold start. Estado final healthy, restart count 0. Houve duas
recriações do TTS para corrigir a continuidade detectada; outros serviços não
foram recriados. Snapshot após smoke: container ~4,857 GiB, CPU 16,85%, GPU
10.674 MiB com utilização 31% naquele instante (tráfego de produção presente).
Isso não mede tempo de kernel por pedido ou economia de GPU.

**Profiling detalhado desligado ao final:** env `AEGIS_TTS_PROFILE_LATENCY=0` e
arquivo operacional ausente. As métricas leves permanecem. Configuração efetiva:

```text
AEGIS_TTS_INITIAL_SEGMENT=stable
AEGIS_TTS_INITIAL_CODEC_FRAMES=22
AEGIS_TTS_CODEC_BATCH_FRAMES=22
AEGIS_TTS_ONSET_TRIM=conservative-v1
AEGIS_TTS_STREAM_STARTUP_MS=320
AEGIS_TTS_PROFILE_LATENCY=0
```

Repetição finita do corpus principal (diretório privado obrigatório):

```bash
python3 scripts/benchmark-first-audio.py \
  --output-dir /CAMINHO_PRIVADO/benchmark \
  --benchmark-url http://127.0.0.1:8001 --wait-idle-seconds 60 \
  --case time --case math --case long --repetitions 3
python3 scripts/smoke-test.py
```

Não usar esse comando em timer/keepalive. Os pares A/B e snapshots completos
ficam em `/var/tmp/aegis-tts-latency-20261101`, diretório 0700, áudios/manifests
0600. Nenhuma referência, peso, WAV privado ou credencial foi enviada ao Git.

Rollback completo preservado em diretório 0700, fora do Git:
`/var/tmp/aegis-tts-latency-20261101/rollback.sh`.
Ele restaura `.env` e o Compose operacional anterior e recria somente TTS,
sem apagar volumes/pesos/cache. A imagem antiga também está retida como
`aegis-tts-gateway:rollback-20261101-ba1beb9`. Nenhum backend conversacional,
Android, STT, memória, reasoning ou serviço do repositório principal foi alterado.

## Nova voz: entrega deliberadamente preparatória

A restrição posterior do proprietário foi cumprida: nenhum fine-tuning,
geração do corpus vocal por API, dataset completo, download de checkpoint grande ou
implantação de nova voz. Só pesquisa, scripts e testes sintéticos pequenos.

[Preparação da arquitetura](../new-voice-preparation.md): FastPitch+HiFi-GAN foi
avaliado primeiro; o catálogo/base encontrado não comprova PT-BR. A segunda e
única alternativa selecionada para piloto futuro é Piper VITS pt_BR-cadu-medium,
com frontend PT-BR e decoder HiFi-GAN. Seu latente é contínuo; não se reutiliza
o codec Qwen que o proprietário observou perder os efeitos. A preservação do
efeito ainda exige reconstrução real e A/B após autorização. Não se afirma que
um modelo inglês virou PT-BR só por receber transcrições.

[Pipeline de arquivo único](../dataset-single-master.md): piloto rascunho de
112 frases (~14 min estimados), expansão rascunho de 792 (~118 min), splits
agrupados, cache/retomada, limites/custo, montagem de um WAV, manifestos, recorte
monotônico, margens e quarentena. São 47 testes do dataset: 30 do pipeline,
incluindo perdas nos extremos e mudanças mascaradas por padding, e 17 de revisão
em lote/bindings. Um comando gera evidências por ID após a aprovação explícita
da amostragem, sem exigir centenas de edições manuais de JSON. A preparação neural tem 13 testes
sintéticos adicionais. Nenhum deles demonstra qualidade vocal neural.

Só existe a demonstração de tons de 22,8 s em
`/var/tmp/aegis-tts-dataset-20261101/synthetic-demo/master_raw.wav`, com manifesto.
Não existe master vocal real. Campos ausentes e comandos futuros constam da
documentação: provedor/endpoint/modelo/voz, direitos/termos, preço/tetos e aprovação
vinculada ao hash do corpus. Quando autorizado, o proprietário abrirá **um**
`master_raw.wav`, aplicará a receita a tudo e exportará **um**
`master_processed.wav`, preservando a linha do tempo. O script faz os recortes e
validações; não há centenas de operações manuais. Treino futuro depende do master
tratado revisado, pesos autorizados e janela exclusiva de GPU, e usa perfil novo
`AegisVoicev2.0-PiperPTBR-pilot`. WORLD permanece na voz atual até substituta aprovada.
