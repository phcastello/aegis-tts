# Primeiro áudio — experimento v0.8.0

## Estado e evidência

Em 31/10/2026 o endpoint privado `http://10.1.1.47:8001` estava pronto,
com fila vazia. Seis chamadas reais ao contrato nativo completaram, somando
oito segmentos de modelo; não houve novo erro nem cancelamento. Não foi
iniciado outro modelo/GPU, reiniciada produção ou adicionado tráfego periódico.

O checkout examinado foi `2564fc7`, clonado de `phcastello/aegis-tts`.
Não havia `AGENTS.md` nesse repositório nem nos ancestrais locais.
A origem/imagem/digest/hardware do serviço **não foram comprovados**:
a credencial SSH já configurada no servidor Aegis foi recusada pelo host TTS.
O acesso HTTP permite síntese, readiness e métricas, mas não administração.
Portanto esta branch é uma candidata baseada na fonte disponível, ainda
pendente de comparação com a fonte efetivamente implantada. Não houve deploy.

O `/v1/models` real identifica `/models/Qwen3-TTS-12Hz-0.6B-Base`, com
`max_model_len=4096`. O header de áudio identifica
`AegisVoicev1.0-Q00-W04-P07-M02`. O checkout fixa a imagem base
`vllm/vllm-omni@sha256:8b0cc5438eb27b34cdfd22011b735da6a94835a09e9c56ddf9e8cb300d679919`,
Q00 (`temperature=.05`, `top_p=.3`, `top_k=1`, `repetition_penalty=1.15`,
`seed=40733133`, `max_tokens=1024`, `non_streaming_mode=true`). Isso é
configuração da fonte, não comprovação da imagem implantada. A opção
`non_streaming_mode` foi preservada; não se infere dela ausência de PCM incremental.

## Baseline real

Todos os textos são completos e sintéticos; nenhum dado pessoal ou referência
foi incluído nos resultados versionados. O primeiro observado ocorreu após
ociosidade desconhecida; não é chamado de inicialização fria. Readiness já
estava aquecida. Todas as chamadas começaram com fila vazia.

| Caso | Repetição | HTTP headers (ms) | Primeiro PCM (ms) | Bloco com PCM não silencioso (ms) | EOF (ms) | Áudio (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| horário | 1, primeiro observado | 17,7 | 1.657,5 | 1.658,9 | 2.566,0 | 3,59 |
| matemática | 1 | 17,6 | 1.608,0 | 1.609,2 | 2.447,2 | 3,34 |
| longo | 1 | 17,9 | 1.605,8 | 1.607,3 | 6.432,1 | 9,66 |
| horário | 2 | 17,8 | 1.613,0 | 1.614,3 | 2.094,1 | 2,95 |
| matemática | 2 | 17,6 | 1.612,8 | 1.614,7 | 2.490,4 | 3,34 |
| longo | 2 | 19,2 | 1.626,5 | 1.627,9 | 6.627,0 | 9,82 |

As seis amostras têm primeiro PCM médio de 1.620,6 ms, mediana de
1.612,9 ms e máximo de 1.657,5 ms. O bloco não silencioso médio chega em
1.622,1 ms. O limiar de inspeção é `abs(PCM16)>100`; não representa uma
medição acústica. O primeiro bloco HTTP/PCM pode conter silêncio inicial:
registramos os dois marcos separadamente. O download já se sobrepõe à geração.

O corpus fixo consta de `scripts/benchmark-first-audio.py`. Os resultados
individuais estão em [tts-direct-baseline.json](tts-direct-baseline.json).
Os PCM/chunks/IDs sintéticos completos permanecem privadamente no servidor
Aegis, fora do Git. O serviço local não informa custo monetário por síntese;
sem acesso administrativo não foi possível atribuir CPU/GPU de produção.

Não foram medidos aqui API Aegis, proxy habitual, rede móvel, Android,
endpointing ou alto-falante. Estes números não demonstram a meta de quatro
segundos de fim de fala até resposta útil.

## Mudanças candidatas

O caminho nativo `_segment_adapter` não atualizava as métricas de DSP:
elas permaneciam em zero apesar das sínteses reais. O profiling do helper
legado `_processed_stream` não media o caminho efetivo.

A candidata mede, sem texto nem áudio nos eventos: espera da fila, headers
upstream, primeiro PCM bruto, início/fim do primeiro processamento, quantidade
de áudio acumulado, primeiro PCM processado, bloco não silencioso, EOF e custo
de WORLD/DSP. Eventos têm `elapsed_ms` monotônico local e o ID de requisição
existente. Tempos de outro processo não são subtraídos. O profiling detalhado
é opt-in (`AEGIS_TTS_PROFILE_LATENCY=1`); o padrão cotidiano é `0`.
As métricas agregadas leves continuam disponíveis. Readiness e áudio informam
SHA de build (quando fornecido) e variante em headers, sem credenciais.

WORLD e M02 são executados em um worker sequencial, fora da tarefa do event loop.
A mesma instância de estado não é usada em paralelo. Peças internas de no
máximo 250 ms limitam trabalho por operação mesmo quando o transporte entrega
um bloco muito grande; elas não definem fronteiras acústicas. Cancelamento
aguarda a operação CPU já iniciada e descarta saída cancelada. O buffer/fila
e o controle de fluxo existentes permanecem limitados.

O padrão `stable` mantém target de 1.250 ms, lookahead de 200 ms e raio de
busca de 300 ms: precisa de aproximadamente 1.750 ms **de áudio**, não de
relógio, antes da primeira renderização normal. O encerramento pode liberar
um trecho menor. A variante opt-in `initial-quiet-900ms-v1` procura uma
fronteira quieta entre 750 e 1.050 ms, com target de 900 ms e os mesmos 200 ms
de contexto/lookahead. Assim pode decidir após 1.250 ms de áudio. Sem fronteira
quieta aguarda e utiliza a janela estável; não força uma nova emenda dentro de
uma palavra. Segmentos posteriores mantêm os parâmetros estáveis.

Uma variante preliminar que forçava cortes menores foi rejeitada: adicionou
uma emenda de 10 ms ao material matemático sintético. Não foi publicada nem
implantada. A candidata conservadora continua desativada até avaliação da voz
real. Nenhum asset, hash, modelo, referência, transcrição, normalização de
pronúncia, WORLD/P07/M02 ou velocidade de reprodução foi alterado.

## Verificação e limitações acústicas

No servidor Aegis foram usados Python 3.12.15, numpy 1.26.4, scipy 1.13.1 e
pyworld 0.3.4 em venv privado. CPU local: Xeon E5-2640 v4, oito CPUs lógicas
visíveis. Esse é o host do experimento CPU, não o hardware TTS remoto.
Foi necessário `setuptools<81` no venv local porque pyworld 0.3.4 importa
`pkg_resources`; as dependências de produção não foram alteradas.

Os testes cobrem fragmentação arbitrária inclusive amostra PCM dividida,
finalização curta, cancelamento, progressão do event loop com worker,
PCM antes de EOF, estado contínuo e escolha conservadora de fronteira.
A comparação do padrão estável com `voice_pipeline.py` original foi
**bit a bit idêntica** nos três PCM sintéticos espeak. Isso é evidência de
mecânica do DSP, não homologação da identidade da voz Qwen.

A comparação local no mesmo PCM espeak está em
[dsp-mechanics.json](dsp-mechanics.json). A variante conservadora preservou as
durações e contagens de emendas estáveis nesses três casos. O longo tem 20 ms a menos que o raw nos dois caminhos por duas emendas
preexistentes; isso não foi ocultado nem atribuído à otimização. Não houve clipping
digital (pico de 29.205 PCM16); o caso horário permitiu antecipar a renderização
com 1.250 ms acumulados, os outros caíram em 1.750 ms. Tempos CPU isolados
variaram com atividade de desenvolvimento/primeiro import e não sustentam
uma alegação de ganho de produção.

Um experimento CPU com WORLD real e o mesmo PCM matemático mediu intervalos
máximos do event loop de 655/1.487 ms com DSP síncrono, contra 133/137 ms com
worker (duas observações por modo). Extensões C ainda podem segurar o GIL:
isto reduz a espera pela renderização inteira, sem garantir responsividade
de áudio em tempo real nem reduzir por si só a latência de inferência. O
observador temporário de 10 ms existiu só nesse processo de benchmark.

Não houve capacidade de escuta humana neste experimento. Não se comprovou
preservação auditiva de primeiras consoantes, pitch/formantes e pronúncia na
voz Qwen. A validação de integridade do perfil não substitui essa avaliação.
Por isso nenhuma variante acústica foi promovida.

## Reprodução, implantação e rollback

Resultado local: `42 passed, 1 skipped` (GPU opt-in não executado);
`validate-profile`, `validate-lexicon`, `run-pronunciation-corpus`,
`run-golden-regression` e `docker compose config --quiet` passaram.
`run-golden-regression` comprovou integridade do perfil, sem escuta auditiva.

Testes locais:

```bash
python3 -m pytest -q
docker compose config --quiet
```

Para completar o experimento com acesso administrativo ao host TTS, primeiro
registre checkout efetivo/dirty state, labels Compose, imagem/digest, versão
de vLLM-Omni, CPU/GPU e ocupação. Compare `gateway/` da imagem com a fonte;
derive ou reconcilie a branch contra aquela fonte antes de compilar. Preserve
a imagem anterior por digest/tag e a configuração operacional fora do Git.
Nunca copie referências ou tokens para um relatório.

`benchmark-first-audio.py --capture` deve rodar **dentro do container existente**,
com `/gateway` no PYTHONPATH e o mount protegido original disponível. Ele faz
somente três chamadas ao modelo já carregado e guarda raw PCM sintético e
tempos de chunks num diretório privado explícito. `--replay` usa exatamente o
mesmo PCM e a mesma agenda de chegada para as duas variantes; não chama o
modelo novamente. Copie o script para `/tmp` ou execute-o pelo stdin, e use
um destino privado como `/tmp/aegis-latency-v080`. Não crie outro container GPU.
O arquivo não cria endpoint público para texto arbitrário ou coleta cotidiana.

```bash
python3 /tmp/benchmark-first-audio.py --output-dir /tmp/aegis-latency-v080 --capture --replay
```

O procedimento operacional continua sendo build + `docker compose up -d`,
com os mesmos projeto, arquivos, mounts, volumes, modelo e referência. Para
candidata validada, use uma tag de imagem específica e informe
`AEGIS_TTS_SOURCE_COMMIT=<SHA completo>` no build. Primeiro disponibilize
`AEGIS_TTS_INITIAL_SEGMENT=stable`, habilitando profiling só na janela de
benchmark. Readiness faz o warm-up legítimo já existente; não há keep-alive.
Confirme saúde, readiness, headers, cancelamento e síntese real antes de
considerar a troca concluída. Desligue o profiling depois das medições.

A variante `initial-quiet-900ms-v1` só deve ser selecionada após A/B com raw
Qwen e verificação auditiva, conservando o resultado e a configuração como
variante explícita. Para voltar à segmentação anterior, selecione `stable`.
Para rollback integral use a imagem anterior registrada com os mesmos mounts
e Compose, sem reconstrução de pesos, remoção de volumes ou edição do perfil.
`publish-dev` da Aegis não implanta este serviço independente.


## Continuação no host TTS — 31/10/2026

Esta seção preserva a investigação anterior e registra a verificação administrativa
no servidor real. A fonte operacional foi reconciliada antes das alterações:
os 11 arquivos Python/shell/requirements de `/gateway` na imagem coincidem byte
a byte com `2564fc7` (e `63dff1f`, cuja árvore de runtime é igual).
O checkout inicial estava limpo, em `agent/version-protected-assets`, HEAD
`63dff1f`. A imagem não registra um SHA Aegis de build; seu label de revisão
pertence ao vLLM. Portanto a afirmação comprovada é equivalência dos arquivos
de runtime, não identificação histórica do commit usado pelo build.
A branch de trabalho deriva de `d3b4c07` e preserva a candidata preparada.

O host usa Ubuntu 24.04.4, kernel 6.8.0-139, 12 CPUs lógicas Xeon E5-2470 v2,
23,47 GiB de RAM e RTX A2000 12 GB (11.514 MiB visíveis), driver 535.309.01.
O driver anuncia CUDA 12.2; a imagem usa CUDA 13.0.2, Torch 2.11.0+cu130 e
compatibilidade CUDA já habilitada. vLLM e vLLM-Omni instalados são 0.22.0.
O modelo continua Qwen3-TTS-12Hz-0.6B-Base. A inspeção sanitizada, hashes das
fontes instaladas e parâmetros numéricos efetivos constam de
[host-runtime-20261031.json](host-runtime-20261031.json).

A produção inicial era o container `924896d8b297`, projeto/serviço Compose
`aegis-tts`, imagem `sha256:6174bb849eea90fbe2fff51153a70cb3489585c7c052b4fed404a9239278f491`.
O digest base continua `8b0cc5438eb27b34cdfd22011b735da6a94835a09e9c56ddf9e8cb300d679919`.
Mounts de modelo e perfil são somente leitura; cache é o mesmo bind persistente.
Em repouso: fila vazia, GPU 0%, 10.676 MiB ocupados; aproximadamente 4,895 GiB
no container e 16,3% CPU no snapshot Docker (100% corresponde a um núcleo).
A capacidade continua limitada a uma síntese ativa e um item por estágio GPU.
Não foi criada outra instância de modelo nem outro container GPU.

### Gargalo confirmado

Nove chamadas nativas sem alteração, três repetições de cada texto, mediram
primeiro PCM médio de 1.646,1 ms, mediana 1.639,8 ms e máximo 1.691,3 ms.
A primeira observação foi 1.674,8 ms, sem reinicialização deliberada.
Headers chegaram em média a 16,7 ms. Todas começaram com fila vazia.

Uma instrumentação temporária em outro gateway CPU, apenas no loopback do
mesmo container e usando o mesmo backend carregado, capturou o caminho original
sem modificar produção. Os quatro segmentos reais mostraram primeiro bruto
entre 316,9 e 375,3 ms; acumulação até o primeiro WORLD entre 1.359,5 e
1.537,8 ms; primeiro WORLD entre 218,2 e 292,9 ms. O primeiro PCM processado
ocorreu entre 1.596,2 e 1.789,4 ms no relógio local do adapter. A primeira
observação desse processo inclui caches/imports locais e foi mantida nos dados.
Essas chamadas diagnósticas não substituem o baseline HTTP de produção.

O primeiro chunk contém 1 frame = 1.920 samples = 80 ms de áudio.
O próximo lote contém 25 frames = 2.000 ms de áudio. A primeira decisão stable
necessita 1.750 ms de áudio, mas acaba recebendo aproximadamente 2.080 ms na
segunda chegada. O tempo dominante é esperar essa chegada, cerca de 1,04–1,16 s
após o primeiro chunk; não é esperar 1,75 s de relógio no WORLD.
Fragmentos HTTP desse lote podem dividir os 96.000 bytes; não são novos chunks
de codec. A candidata pode contabilizar 1.945,17 ms recebidos ao iniciar DSP
porque alimenta o lote em peças de até 250 ms, sem mudar a decisão acústica.

O chunk inicial é quase silencioso (picos 4–12 PCM16 nas nove capturas), abaixo
do limiar `abs(PCM16)>100`. O primeiro bloco bruto não silencioso só chega com
o segundo lote. Há também aproximadamente 0,43–0,59 s de silêncio inicial no
próprio áudio gerado. A chegada de um bloco contendo voz não é o instante em
que um alto-falante já reproduziu a primeira consoante. O benchmark registra
esse offset e uma estimativa de reprodução imediata, sem confundi-la com uma
medição Android ou com a meta global de quatro segundos.

A configuração já habilita async chunking, prefix caching, cache de referência
e embedding de locutor, primeiro chunk mínimo de um frame e CUDA Graph no
Code2Wav. Não há um streaming desabilitado a simplesmente ligar. Os caches de
referência são reutilizados pelo runtime após warmup; a serialização HTTP da
referência ainda se repetia em cada segmento.

Há uma divergência entre intenção e API instalada: o gateway envia
`non_streaming_mode=true`, mas o schema desta imagem descarta esse campo;
o builder Base usa o padrão `false`. Os controles amostrais Q00 são preservados
pelo YAML de deploy, inclusive os campos top-level ignorados pela API.
Esta tarefa preserva o condicionamento efetivo e não corrige esse descompasso.
A descrição anterior do flag era baseada na fonte, não na implementação real.

Não foi possível separar início do kernel de inferência, primeiro codec e
primeira decodificação dentro dos processos já carregados: a API e `/metrics`
desta imagem não expõem esses marcos por ID. O intervalo até o primeiro bruto
é medido; suas parcelas internas permanecem não medidas. Não se deduziu tempo
GPU a partir de duração de áudio, nem se subtraíram relógios entre processos.
Os adapters usam perf_counter local; eventos de fila usam monotonic local com
o mesmo ID. Utilização amostrada de GPU é diferente de tempo de kernel por pedido.

### Otimizações conservadoras e candidatos acústicos

O worker sequencial conserva o estado acústico, processa uma operação por vez,
aguarda trabalho CPU iniciado no cancelamento e descarta saída cancelada.
Uma comparação com os mesmos PCM e tempos de chegada, com cliente HTTP em outra
thread, mediu picos de `/health` de 496–1.201 ms no original, contra 16–39 ms
no worker. Os maiores intervalos do event loop caíram de 540–1.246 ms para
130–164 ms. Extensões C ainda retêm o GIL por alguns períodos. Esses números
medem responsividade durante replay CPU, não capacidade concorrente do modelo.
O hash de saída coincidiu entre original, candidata síncrona e worker em todos
os casos desse ensaio (matemática/longo, duas repetições).

O detector M02 agora usa doubles nativos na mesma ordem de operações.
Nos quatro segmentos Qwen, duas repetições, o primeiro mastering passou de
19,20 para 7,65 ms; o primeiro DSP completo, de 276,07 para 265,98 ms em replay
CPU sem espera. CPU total médio dos segmentos passou de 942,57 para 915,57 ms.
São aproximadamente 10 ms potenciais no primeiro áudio, não uma redução de
centenas de milissegundos. A saída float64/estado e o PCM completo são idênticos.

Os campos invariantes da referência são serializados uma vez no startup;
somente o texto sintético é serializado por segmento. O payload JSON efetivo
permanece igual, incluindo escape de Unicode, aspas e tentativas de injeção.
Um ensaio isolado com referência sintética de 2,5 MB, 100 iterações, caiu de
21,28 para 1,75 ms médios, com custo persistente aproximado de 2,5 MB.
A instrumentação real separa essa preparação da espera upstream.
Nenhum conteúdo da referência é registrado. `protected/` fica fora do contexto
Docker, e continua disponível exclusivamente pelo mount operacional.

O replay temporizado das nove capturas reais, incluindo EOF e importação de
pyworld antes dos cronômetros, confirmou PCM original/stable bit a bit igual,
independência da fragmentação, nenhum clipping e durações preservadas. As
emendas estáveis existentes podem reduzir 10–20 ms frente ao raw; não foram
introduzidas pela otimização. `initial-quiet-900ms-v1` fez fallback e produziu
o mesmo PCM nas nove amostras. Sua média temporizada foi 1.635,7 ms contra
1.634,8 ms stable; diferenças de CPU/OS sem mudança de decisão não são ganho.
A variante continua desativada, pois outros textos podem ativar nova fronteira.

O override autorizado `initial_codec_chunk_frames=22` foi testado em nove
capturas separadas, sem mudar a configuração persistente. Ele entrega 1.760 ms
de áudio na primeira chegada, perto de 1,08 s, suficiente para a janela stable.
O PCM bruto mudou: nos casos curtos houve diferenças de até 6–7 unidades PCM16;
no longo houve diferenças maiores e uma geração com duração diferente entre
coortes. Gerações independentes não isolam toda variância do modelo; não há
prova de equivalência perceptiva. Esse ajuste não pode ser promovido com base
apenas nas métricas. WAVs identificados para revisão ficam exclusivamente no
diretório privado operacional. A referência, modelo, transcrição, léxico,
normalização, W04/P07/M02 e velocidade de reprodução foram preservados.

### Estado operacional desta revisão

As alterações de código ainda aguardam os checks finais e a implantação
controlada. O registro após implantação será acrescentado abaixo; build sozinho
não será considerado atualização da produção. A imagem original foi retida
como `aegis-tts-gateway:rollback-20261031-6174bb8`. Snapshot completo do Compose,
inspeção, WAVs e script de rollback ficam em `/var/tmp/aegis-tts-latency-20261031`,
modo 0700, fora de qualquer checkout Git; arquivos de áudio têm modo 0600.
O rollback usa o Compose anterior com os mesmos mounts e projeto, sem `down`,
remoção de volumes ou alteração de permissões protegidas.

### Validação antes da troca de imagem

A suíte local passou com **53 passed, 1 skipped** (GPU opt-in ainda não
executado nesta etapa). Perfil, léxico, corpus de pronúncia, golden regression,
Compose e revisão independente dos contratos passaram. Golden regression
existente verifica integridade do perfil; não substitui escuta humana.

O ensaio de chunk inicial22 + DSP stable mediu primeiro processado médio de
1.333,3 ms, mediana 1.327,0 ms, máximo 1.399,5 ms em replay com cronograma real.
O ganho potencial frente ao replay default foi 301,5 ms. A versão muda PCM e
continua bloqueada para promoção por falta de revisão auditiva do proprietário;
não há evidência de áudio útil abaixo de um segundo com a voz preservada.

O gateway CPU com a candidata efetiva, compartilhando somente o backend já
carregado, mediu nove chamadas: média 1.641,5 ms, mediana 1.639,7 ms,
máximo 1.711,2 ms. A primeira foi mantida. Não há redução substancial comprovada
nessa coorte pequena frente a 1.646,1 ms em produção original; o benefício mais
forte demonstrado é responsividade do worker, com microganhos CPU/serialização.
Preparação real do payload caiu de aproximadamente 18 ms para 0,75–2,38 ms.
O primeiro bruto dos primeiros segmentos ficou em 299,7–346,2 ms e o primeiro
DSP consumiu 222,5–292,8 ms de CPU por thread.

Um ensaio intermediário de loopback importou o gateway original por causa do
working directory do container. Foi identificado pelo header de revisão ausente,
retido separadamente nos resultados e repetido com working directory explícito
e header `predeployment-candidate` verificado. Não foi chamado de candidata nem
usado como evidência de implantação.

Os dados individuais sanitizados, inclusive chamadas lentas, traces, coortes
intermediárias e resultados dos replays, estão em
[host-benchmark-20261031.json](host-benchmark-20261031.json). Raw direto e
replay usam o texto inteiro do corpus; a API nativa segmenta o caso longo.
EOF e duração dessas duas modalidades não devem ser comparados como se fossem
a mesma requisição. O benefício de22 frames usa apenas as coortes de captura
inteira com a mesma política DSP; a variância entre gerações está preservada.

Para repetir, execute o benchmark no container existente com a fonte candidata
como diretório de trabalho/PYTHONPATH, confirme o header de revisão quando usar
um gateway de teste, e mantenha saídas fora de Git:

```bash
python3 scripts/inspect-runtime.py
python3 scripts/benchmark-first-audio.py --output-dir PRIVATE --capture --repetitions 3
python3 scripts/benchmark-first-audio.py --output-dir PRIVATE --replay --baseline-pipeline ORIGINAL/voice_pipeline.py
# Experimento separado, nunca aplicado ao Compose por este script:
python3 scripts/benchmark-first-audio.py --output-dir PRIVATE22 --capture --repetitions 3 --initial-codec-chunk-frames 22
python3 scripts/profile-dsp.py --input-dir PRIVATE --output PRIVATE/profile.json --baseline-pipeline ORIGINAL/voice_pipeline.py
```

A promoção conservadora se apoia em PCM idêntico, microganhos mensuráveis e
melhora comprovada na estabilidade/responsividade, conforme a política de
promoção da tarefa. Não implica ter atingido a meta de primeiro áudio útil em
um segundo nem a meta global da Aegis.
