# Integração experimental VoxCPM2 no gateway

O operador concluiu seis provas HTTP com checkpoint400 de engenharia e encerrou
os servidores. A [política validada](voxcpm-gateway-execution-20261103.md) inclui
CacheGuard, `ptbr-numbers-v1` e reserva320ms; os42 requests com reserva foram
completos e sem lacunas. O treino LoRA continua separado. Identidade e
seleção final permanecem abertas. Tempos do motor nativo não são tempos deste
gateway: normalização, segmentação, pausas entre segmentos e transporte mudam
o percurso medido.

`AEGIS_TTS_BACKEND=qwen` permanece o padrão. Compose, Dockerfile, entrypoint,
dependências da imagem e o perfil protegido v1 não mudaram. A feature flag
`AEGIS_TTS_BACKEND=voxcpm2` exige `AEGIS_VOXCPM_CONFIG`, arquivo JSON local do
operador. Clientes HTTP não escolhem modelos, referência, LoRA, seed ou DSP.
Importar o módulo opcional não importa Torch; o carregamento ocorre somente
após selecionar explicitamente esse backend e iniciar seu lifespan.

O perfil experimental é `AegisVoicev2.0-VoxCPM2-experimental` na API nativa.
O header completo identifica
`AegisVoicev2.0-VoxCPM2-LoRA-r8-N10-Passthrough-OriginalNFC-Experimental`, ou uma variante
`W04-P07-M02` explicitamente configurada. Pedir o perfil v1 nessa instância é
rejeitado. `X-Aegis-TTS-DSP-Variant=passthrough` declara o bypass de WORLD;
o áudio-alvo robótico de treino não prova que uma saída nova reproduza sua
identidade. O passthrough faz somente FIR e quantização PCM, sem ganho/limiter.
O modo `world-w04-p07-m02` chama o DSP existente e declara sua aplicação no
header. O corte de silêncio inicial continua uma opção separada, identificada
no header de runtime; deve ser registrado no benchmark.

## Arquitetura e isolamento

`voxcpm_runtime.VoxBackend` mantém uma única thread durante a vida da instância.
Ela carrega, aplica LoRA, move pesos/caches à GPU, otimiza, executa três chamadas
de aquecimento completas, avança/fecha cada gerador e limpa KV/AudioVAE.
Readiness permanece 503 até os três aquecimentos produzirem PCM completo. A
primeira chamada fria/compilação é registrada separadamente das duas seguintes
em `backend.warmup`, com ID da thread e bytes. O serviço aquece somente essa
frase fora do corpus. A primeira requisição inédita após readiness deve ser
medida sem aquecimento por caso. O benchmark nativo anterior aquecia cada caso
três vezes: seus 192–412 ms não representam automaticamente este uso do gateway.
`/metrics` registra carga/readiness, picos de RAM/GPU/cache e contadores de
compilação; novas grafias/comprimentos podem recompilar e aumentar a variância.
O startup frio fica separado das requisições após readiness. O probe habilita
`instrument_replays:true`: conta chamadas reais de `CUDAGraph.replay` com um
wrapper temporário que chama a função original e a restaura no fechamento.
O modo comum deixa essa instrumentação desligada; wrappers presentes, sozinhos,
não confirmam replay. Eventos SSE de perfil registram contadores por segmento
quando `AEGIS_TTS_PROFILE_LATENCY=1`, sem texto ou caminhos privados.

A carga usa o backbone BF16, AudioVAE FP32 e os ramos LoRA originais em BF16,
após cópia estrita dos parâmetros FP32 salvos. Não mescla o adapter nem altera
os pesos de origem. Base, VAE, parâmetros, buffers, tensores ordinários e ambos
os caches KV devem residir na GPU; não há offload na inferência. Há lock local
da instância e verificação de GPU inicialmente livre. O servidor deve usar um
único worker. Não é seguro executar este teste junto a Qwen ou treinos nessa GPU.

O adapter é um checkpoint local confiável com `.sha256.json`, SHA fixo no
JSON e configuração rank/source/model coincidentes. Hash e desserialização
usam o mesmo arquivo aberto, preservando o inode diante de gravação atômica.
Não aponte o runtime para `last.pt` sendo atualizado: selecione um checkpoint
estável com sidecar válido. O loader upstream permite faltar LoRA na base;
o adaptador exige que somente as chaves LoRA faltem e que nenhuma chave extra
ou outro peso ausente seja tolerado. Valida o estado LoRA finito, FP32, conjunto
de nomes e shapes exatos antes de converter sua cópia de inferência.

Fonte externa limpa pinada em
[`cce58a5`](https://github.com/OpenBMB/VoxCPM/tree/cce58a5b59303c9bd63d12f23afe6a49a4a80c59),
pesos em
[`32279ef`](https://huggingface.co/openbmb/VoxCPM2/tree/32279effe8c19989596f05d353d1447f51d9e915).
Código/pesos Apache-2.0; corpus/voz privados e direitos do adapter permanecem
separados. Não há upload de pesos ou áudio. Hashes fixos:

| Arquivo | Bytes | SHA256 |
| --- | ---: | --- |
| model.safetensors | 4580080592 | f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d |
| audiovae.pth | 376951122 | 94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1 |

KV1024, max_len400, NFE10, CFG2, prefix4, seed40733133, sem referências de
clonagem, VAD ou retry. Configuração e tokenizer usam blobs pinados. A
otimização oficial `reduce-overhead` é obrigatória, com os quatro wrappers
verificados; três chamadas na mesma thread antecedem readiness. Os
[recibos HTTP](voxcpm-gateway-execution-20261103.md) registram replays reais e
verificações de residência no gate GPU. Não importar helpers de treino evita executar packers, loops, sinais
ou supervisores ao servir áudio.

`voxcpm_audio.HalfRateFIR` copia inalterado o resampler validado: 41 taps,
Kaiser10, lookahead20 amostras nativas, fase global, compensação exata do atraso
e `ceil(N/2)` amostras na saída. Guarda até41 amostras de histórico. Fecha e
flusha somente no EOF normal; cancelamento descarta a cauda pendente. NaN/Inf,
canais errados ou overshoot PCM geram falha sem clipping silencioso. O gerador
é fechado antes de zerar os caches; os endereços KV são preservados para graphs.
Uma bridge de dois blocos limita prefetch. Buffer cheio aplica backpressure e
timeout10s; cancelamento cooperativo aguarda o passo corrente, fecha o gerador
na thread proprietária, descarta FIR e libera a próxima síntese depois da limpeza.

## Texto e HTTP

Frontend experimental padrão: `original-nfc`, mantendo grafia, maiúsculas, números,
siglas e Aegis. Não importa o léxico protegido. Mantém
o limite12000 caracteres e a segmentação de frases com limite320 caracteres.
Um segmentador experimental restaura tokens técnicos em cada frase (inclui
decimais, URLs, IPs e emails); o segmentador Qwen não foi modificado.

O controle explícito `frontend:f5-refinement-1` usa uma cópia auditada da função
leve `training/f5_refine_text.py`, sem importar o pacote de treino no runtime.
Expande termos técnicos, números/horários e o nome Aegis, segundo a mesma receita
F5 já documentada. Versão/hash da origem e equivalência das sete frases são
verificados por teste e recibo. O perfil troca `OriginalNFC` por `F5Refinement1`;
o header `X-Aegis-TTS-Frontend` e o manifest identificam a escolha. Treinador,
transcrições e padrão original permanecem intactos. A motivação são divergências
de ASR em horários/siglas; isso não confirma um erro auditivo do TTS nem aprova
automaticamente a expansão. Compare as variantes separadamente, incluindo
esperados literal/falado/efetivo distintos no ASR.

Antes de enfileirar ou enviar headers, conta os tokens efetivos de todos os
segmentos e exige `tokens + 400 + 2 <= 1024`. Excesso retorna413 com código
`BACKEND_TEXT_TOO_LONG`; não encurta texto. Um gerador que alcance o teto
autoregressivo efetivo é tratado como conclusão incerta e falha. Isso não
confirma semanticamente uma saída que termine antes do teto: ASR e escuta
continuam necessários. Se a falha ocorrer durante streaming, bytes já recebidos
podem ser parciais e o status/SSE marcará `failed`; WAV completo responde502.

As rotas de saúde, prontidão, métricas, fila, status, SSE e DELETE reutilizam o
contrato existente. WAV somente com `stream:false`; contém RIFF completo,
PCM s16le mono24k. Streaming somente `stream:true,response_format:pcm`, sem
cabeçalhos WAV. `stream:true,response_format:wav` retorna422 nesta variante.

## Teste futuro isolado na porta8002

O exemplo [`config/speech/voxcpm2-experimental.example.json`](../../config/speech/voxcpm2-experimental.example.json)
contém placeholders de caminhos e hash zero: substituí-los em arquivo privado
é necessário antes de iniciar. Reuse o ambiente existente compatível com
Torch/Transformers/Vox e os headers do compilador já auditados; não há instalação
ou imagem nova nesta receita. `compiler_prerequisites` opcional aceita o recibo
de headers e somente as variáveis CPATH/CC/CXX. Cache compilado limitado a512MiB,
com5GiB livres; exceder o orçamento causa falha sem remover
arquivos ativos. A versão atual faz esses checks em um único CacheGuard
periódico5s, com snapshots e idade máxima15s; não percorre diretórios em cada
geração. A carga CPU inicial exige13GB de RAM disponível medidos.

Com GPU liberada pelo operador e checkpoint selecionado:

```bash
AEGIS_TTS_BACKEND=voxcpm2 \
AEGIS_VOXCPM_CONFIG=/CAMINHO/PRIVADO/voxcpm2-runtime.json \
AEGIS_TTS_ONSET_TRIM=off AEGIS_TTS_STREAM_STARTUP_MS=320 \
PYTHONPATH=gateway \
/CAMINHO/AMBIENTE/bin/python -m uvicorn default_reference_gateway:app \
  --host 127.0.0.1 --port 8002 --workers 1
curl -f http://127.0.0.1:8002/health
curl -f http://127.0.0.1:8002/ready
python3 scripts/smoke-test.py --base-url http://127.0.0.1:8002
```

Os comandos reais das provas ficam nos manifests do executor. O gate de
engenharia confirmou cold+warm, WAV/PCM, cliente, cancelamento e nova síntese,
saúde/readiness. O próximo gate repete a medição com os pesos finais salvos,
incluindo ASR, comparação com Qwen e avaliação de identidade. Restaurar Qwen
usa seu Compose/entrypoint original sem a feature flag; nenhuma configuração
protegida ou padrão foi alterado por esta preparação.

## Executor da janela GPU preparado

`training/v2_voxcpm_gateway_probe.py` é um plano por padrão. `execute` recusa
execução fora do GPU worker exclusivo, com ASR pausado e CPU ociosa. Foi
executado nas provas documentadas. Recebe caminho e SHA do checkpoint selecionado,
verifica seu sidecar e exige espaço de trabalho abaixo de 40 GB, mais margem
de 1 GiB para este experimento e reserva livre de 5 GiB. Não salva checkpoints.
Inicia somente seu filho uvicorn8002 e o encerra em `finally`, conferindo PID,
start ticks e UID antes de sinalizar. Não inicia ou restaura Qwen.

A prova real de imports no mesmo Python F5 passou sem construir modelo e sem
CUDA: Torch2.6.0+cu118, Pydantic2.14.2 e Transformers4.49 continuam no ambiente
F5. Um overlay privado de 1,39 MB fornece somente FastAPI0.115.12,
Starlette0.46.2 e Uvicorn0.34.0; os três wheels somam 229.516 bytes, com hashes
fixos e arquivos extraídos conferidos. Não usa o `site-packages` amplo do host
nem pip. `simplejson`/`sortedcontainers` não estão disponíveis nesse ambiente e
não são importados pelo caminho real de inferência verificado; não foi instalada
a árvore inteira de extras do upstream. Recibo real:
`/var/tmp/aegis-v2-work/voxcpm-gateway/preparation/cpu-import-check.json`.

```bash
python3 -m training.v2_voxcpm_gateway_probe --action prepare-overlay
python3 -m training.v2_voxcpm_gateway_probe --action cpu-check
python3 -m training.v2_voxcpm_gateway_probe
# SOMENTE como job do GPU worker exclusivo, com GPU liberada:
python3 -m training.v2_voxcpm_gateway_probe --action execute \
  --startup-ms 320 \
  --run /CAMINHO/PRIVADO/NOVA-EXECUCAO \
  --checkpoint /CAMINHO/PRIVADO/CHECKPOINT-ESTAVEL.pt \
  --checkpoint-sha256 SHA256-DO-CHECKPOINT
# Controle numérico, depois de concluir/encerrar o primeiro servidor:
python3 -m training.v2_voxcpm_gateway_probe --action execute \
  --frontend ptbr-numbers-v1 --startup-ms 320 --run /CAMINHO/PRIVADO/OUTRA-EXECUCAO \
  --checkpoint /CAMINHO/PRIVADO/CHECKPOINT-ESTAVEL.pt \
  --checkpoint-sha256 SHA256-DO-CHECKPOINT
```

O executor mede startup/readiness em separado, chama o benchmark comum com as
sete frases de validação e três repetições, amostra VRAM e exige exclusivamente
o PID uvicorn esperado na GPU. Não há warm-up por caso: rotula rep0 como frase
inédita após readiness e rep1/2 como repetidas. Guarda eventos SSE/counters,
métricas antes/depois e jobs ASR rep0 com esperado literal/falado/efetivo
separados. Os contratos extras vêm depois da matriz: WAV completo, primeiro
bloco PCM válido seguido de DELETE, status cancelado, nova síntese e readiness.
Logs do servidor limitados a32 MB, preservando a cauda4 MB. Cache512 MiB e
saída512 MiB ficam sob o monitor local; durante a medição não percorre todos
os volumes, usando a margem declarada e o isolamento do processo. O inventário
completo é repetido ao fim. O corpus final de seis frases não é aberto.
