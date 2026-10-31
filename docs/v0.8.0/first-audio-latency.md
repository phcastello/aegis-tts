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
