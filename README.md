# Aegis TTS

Serviço independente e local de clonagem/síntese da voz da Aegis. A
configuração padrão atual usa `vLLM-Omni 0.22.0` com
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` na RTX A2000 12 GB.

URL padrão na LAN: <http://10.1.1.47:8001>

```text
Aegis API (10.1.1.22)
        │ POST /v1/audio/speech
        ▼
Aegis TTS / vLLM-Omni (10.1.1.47:8001)
        │ modelo Qwen Base + ICL por requisição
        ▼
      WAV PCM 16-bit
```

## Runtime padrão atual

O Compose raiz inicia `aegis-tts` em `8001` com a imagem fixada
`vllm/vllm-omni@sha256:8b0cc5438eb27b34cdfd22011b735da6a94835a09e9c56ddf9e8cb300d679919`.
O preset mantém GPU 0, dois estágios, shared-memory connector, async chunking,
codec streaming e CUDA Graphs. `min_tokens` é fixado em `0`: em vLLM-Omni
0.22.0, valores positivos acionam um bug de índice de EOS do modelo Base e
encerram a requisição com erro CUDA.

O endpoint padrão agora é o OpenAI-compatible `/v1/audio/speech`, não o antigo
`/v1/synthesize`. Cada pedido Base deve incluir `ref_audio` (data URL/base64),
`ref_text`, `task_type: "Base"` e `x_vector_only_mode: false`. A rota de health
é `GET /health`. Para o contrato completo e exemplos com a referência padrão,
consulte [docs/API.md](docs/API.md).

O eager 1.7B anterior foi preservado para rollback, sem remover modelos ou
dados:

```bash
./scripts/restore-eager-1.7b.sh
```

As seções abaixo que descrevem FastAPI, UI e `/v1/synthesize` são documentação
do runtime eager preservado; não descrevem a configuração padrão atual.

## Arquitetura

O único serviço Docker, `aegis-tts`, contém FastAPI, a página HTML local, o
modelo Qwen Base carregado em `cuda:0`, o cache de prompts e as métricas. Os
dados persistentes permanecem fora da imagem:

```text
app/       código FastAPI e UI
config/    catálogo de vozes validado
voices/    referências privadas, montadas somente leitura
models/    pesos do Qwen Base, baixados explicitamente
cache/     cache local do Hugging Face
samples/   WAVs opcionais e saídas do smoke test
```

Durante o startup, o estado progride por `starting`, `loading_model`,
`loading_references` e `ready` (ou `degraded`/`error`). O modelo é carregado
uma única vez. Para cada referência válida, o serviço chama
`create_voice_clone_prompt(..., x_vector_only_mode=False)` e mantém o objeto
em memória; ele não é recalculado a cada requisição nem serializado com
`pickle`.

`giseli-34s` é a voz padrão; `giseli-10s` permanece disponível para comparação.
Solicitações sem `seed` usam `41` por padrão para manter os testes reproduzíveis.

## Privacidade e escopo do modelo

O projeto baixa somente este repositório de modelo:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

O Dockerfile não baixa pesos, e o serviço roda com
`HF_HUB_OFFLINE=1` e `TRANSFORMERS_OFFLINE=1`. Assim, uma inicialização nunca
faz download silencioso. O script de download desativa esse modo somente para
a sua invocação explícita.

Referências de voz, transcrições, modelos, caches, samples e `.env` são
ignorados pelo Git e excluídos do contexto de build. Nenhum endpoint retorna
o WAV de referência ou a transcrição. Não coloque referências em `samples/`.

## Pré-requisitos

- Docker Engine e Docker Compose com NVIDIA Container Toolkit;
- driver NVIDIA compatível com CUDA 12.1 (o host atual usa driver 535);
- GPU NVIDIA com VRAM suficiente; a configuração é voltada à RTX A2000 12 GB;
- `curl`, `jq`, `python3` e `sha256sum` no host para os scripts;
- `python3.12-venv` se os testes forem executados em um ambiente virtual local;
- espaço livre suficiente para a imagem, pesos, cache e saídas. Confira antes
  com `df -h` e `docker system df`.

Confirme a GPU dentro de um container antes de iniciar:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.2.2-base-ubuntu22.04 \
  nvidia-smi
```

Não instale drivers NVIDIA no container. As wheels PyTorch CUDA 12.1 são
instaladas na imagem e usam o driver disponibilizado pelo runtime NVIDIA.

## Instalação do zero

```bash
cd ~/aegis-tts
cp .env.example .env
```

Em outro host, ajuste o usuário não-root que escreverá nos bind mounts:

```bash
printf 'AEGIS_TTS_UID=%s\nAEGIS_TTS_GID=%s\n' "$(id -u)" "$(id -g)" >> .env
```

No host atual, os valores padrão `1002:1002` já correspondem ao usuário
`phc`.

### Baixar o modelo

```bash
./scripts/download-model.sh
```

O script mostra o ID e o diretório, mostra o espaço em disco, constrói a imagem
se ela não existir, pede confirmação e baixa somente o Base para:

```text
~/aegis-tts/models/Qwen3-TTS-12Hz-1.7B-Base
```

Ele não remove cache, pesos, imagens, volumes ou outros arquivos. Para uma
automação conscientemente não interativa, use
`AEGIS_TTS_ASSUME_YES=1 ./scripts/download-model.sh`.

### Experimento isolado 0.6B

O serviço de produção continua explicitamente no modelo 1.7B e na porta 8001.
O experimento abaixo cria somente o container `aegis-tts-0.6b`, com a imagem
`aegis-tts:0.6b`, o checkpoint local
`models/Qwen3-TTS-12Hz-0.6B-Base` e a porta local `127.0.0.1:8003`.
Ele reutiliza a mesma referência privada `giseli-34s`, com `seed: 41`, BF16
(com fallback de runtime para FP16) e SDPA; startup permanece offline.

```bash
./scripts/download-model-0.6b.sh
./scripts/benchmark-0.6b.sh
./scripts/compare-models.sh
```

Para automação consciente do download, use apenas:

```bash
AEGIS_TTS_ASSUME_YES=1 ./scripts/download-model-0.6b.sh
```

O benchmark faz um warm-up descartado e cinco gerações sequenciais, salvando
WAVs, headers, métricas e relatórios em `samples/benchmark-0.6b/`. A comparação
gera Markdown, JSON e uma página local com players em `reports/`. Se houver um
golden 1.7B existente, informe-o sem gerar outro:

```bash
AEGIS_TTS_GOLDEN_1_7B=/caminho/para/golden-1.7b.wav ./scripts/compare-models.sh
```

RTF é apenas classificação de velocidade; a qualidade do clone exige avaliação
auditiva humana e o 1.7B não é substituído automaticamente.

### Importar as referências privadas

Coloque à disposição o WAV original e o arquivo de texto cuja transcrição seja
exatamente a fala do WAV. Não invente, não gere e não corrija automaticamente
a transcrição.

```bash
./scripts/import-reference.sh \
  giseli-10s \
  /caminho/giseli-10s.wav \
  /caminho/giseli-10s.txt

./scripts/import-reference.sh \
  giseli-34s \
  /caminho/giseli-34s.wav \
  /caminho/giseli-34s.txt
```

O importador aceita somente os dois IDs configurados, calcula SHA-256, mostra
os metadados técnicos e copia os bytes originais sem conversão. Se um destino
já existir, exige confirmação antes de sobrescrever. Os arquivos de destino
são:

```text
voices/giseli-10s/reference.wav
voices/giseli-10s/reference.txt
voices/giseli-34s/reference.wav
voices/giseli-34s/reference.txt
```

O WAV PCM mono de 32.000 Hz é aceito diretamente. Não o converta para 24 kHz
por conveniência. Se alguma conversão se tornar tecnicamente inevitável no
futuro, preserve o master, gere um derivado com nome distinto, documente o
comando e nunca substitua `reference.wav` silenciosamente.

Valide os dois perfis antes de iniciar:

```bash
./scripts/validate-references.sh
```

O relatório contém codec, sample rate, canais, bits, duração, tamanho, pico,
RMS aproximado, silêncio, clipping, hashes do áudio e texto, e presença da
transcrição. Arquivos fora de 3–60 s, estéreo inesperado, WAV ilegível ou texto
ausente/vazio falham na validação; clipping e silêncio excessivo são
sinalizados sem alterar o áudio.

### Subir o serviço

```bash
docker compose up -d --build
docker compose logs -f aegis-tts
```

O health check do Docker consulta `/health/ready`, com `start_period` de 15
minutos para acomodar o carregamento do modelo e a criação dos dois prompts.
Enquanto isso, `/health/live` e a página de status permanecem acessíveis.

## Configuração

As variáveis começam em `.env.example`.

| Variável | Padrão | Finalidade |
| --- | --- | --- |
| `AEGIS_TTS_PORT` | `8001` | Porta publicada no host. |
| `AEGIS_TTS_INTERNAL_PORT` | `8001` | Porta interna do Uvicorn. Mantenha igual à publicada na configuração usual. |
| `AEGIS_TTS_API_KEY` | vazia | Se definida, exige `X-API-Key` na síntese. |
| `QWEN_DTYPE` | `bfloat16` | O app faz fallback explícito para `float16` se BF16 não for suportado. |
| `AEGIS_TTS_QUEUE_TIMEOUT_SECONDS` | `240` | Limite de espera na fila serial. |
| `AEGIS_TTS_GENERATION_TIMEOUT_SECONDS` | `180` | Limite reportado para uma geração. |
| `AEGIS_TTS_SAVE_SAMPLES` | `false` | Habilita salvar WAV/JSON de respostas. |
| `AEGIS_TTS_MAX_SAMPLES` | `100` | Retenção máxima quando samples estão habilitados. |
| `AEGIS_TTS_CACHE_IMPLEMENTATION` | vazia | Cache KV. Mantenha vazio no serviço normal; `static` é somente para o experimento isolado abaixo. |

`QWEN_DEVICE=cuda:0` e `QWEN_ATTENTION_IMPLEMENTATION=sdpa` são fixados pelo
Compose. Não há FlashAttention nesta primeira etapa.

### Experimento isolado: Static KV Cache

O arquivo `docker-compose.static-cache.yml` cria uma imagem e um contêiner
separados, publicados apenas em `127.0.0.1:8002`. Ele ativa
`use_cache=True`, `cache_implementation="static"` e os logs de graph breaks e
recompilações. Pare o serviço normal antes de iniciar o experimento: os dois
modelos não devem disputar a mesma GPU.

```bash
mkdir -p cache-static samples-static
docker compose stop aegis-tts
docker compose -f docker-compose.static-cache.yml up --build -d
```

Descarte a primeira síntese (compilação). Em seguida, envie três vezes a mesma
requisição, texto e `seed: 41`, registre os headers `X-Aegis-TTS-*`, VRAM e
logs; compare a mediana de RTF e a qualidade. Para encerrar e voltar ao normal:

```bash
docker compose -f docker-compose.static-cache.yml down
docker compose up -d
```

O experimento também pode falhar por incompatibilidade do cache customizado ou
por OOM. Nesses casos, preserve os logs e não habilite a variável no serviço
normal.

## Acesso na LAN e segurança

O processo faz bind em `0.0.0.0`; a publicação padrão é
`0.0.0.0:8001:8001`. Restrinja a porta à LAN solicitada depois de revisar a
regra proposta:

```bash
./scripts/configure-firewall.sh
```

O script mostra a regra e só executa, após digitar `APLICAR`:

```bash
sudo ufw allow from 10.1.1.0/24 to any port 8001 proto tcp
```

Ele se recusa a abrir outra rede, não habilita o UFW e não remove regras. Não
publique a porta na internet. O Compose executa como usuário não-root, remove
capabilities e monta referências/configuração como somente leitura.

Quando `AEGIS_TTS_API_KEY` estiver configurada, envie-a apenas no header
`X-API-Key`. A página web permite informar a chave apenas na sessão do
navegador; ela não é gravada no servidor. A comparação é feita em tempo
constante e a chave não é registrada nos logs. Health checks permanecem sem
autenticação. Não use CORS `*`; a lista de origens conhecidas é configurável
em `AEGIS_TTS_ALLOWED_ORIGINS`.

## Interface e health checks

Abra <http://10.1.1.47:8001/> para a página mínima de teste. Ela mostra estado
do modelo/GPU, voz padrão, texto, seed, parâmetros avançados, player, métricas
e erros. O botão de comparação gera primeiro `giseli-10s` e depois
`giseli-34s`, com mesmo texto/seed/parâmetros; nunca dispara as duas inferências
simultaneamente.

| Endpoint | Semântica |
| --- | --- |
| `GET /` | Página web local de teste. |
| `GET /health/live` | `200` se o processo FastAPI está vivo. |
| `GET /health/ready` | `200` só com CUDA, modelo, voz padrão e prompt padrão prontos; caso contrário `503`. |
| `GET /health` | Estado, dispositivo, dtype, GPU, vozes, tempos e erros sem stack trace. |
| `GET /v1/info` | Informações estáveis do serviço/modelo. |
| `GET /v1/voices` | Metadados seguros de vozes, sem áudio ou transcrição. |
| `POST /v1/synthesize` | Retorna WAV. |
| `GET /docs` | OpenAPI/Swagger local. |

Exemplos rápidos:

```bash
curl http://10.1.1.47:8001/health/live
curl http://10.1.1.47:8001/health/ready
curl http://10.1.1.47:8001/v1/voices
```

## API de síntese

`POST /v1/synthesize` aceita no máximo 1.000 caracteres. `language` aceita
`pt-BR` e `Portuguese`, ambos normalizados para `Portuguese`. Os únicos campos
de `generation_parameters` permitidos são `temperature`, `top_p`, `do_sample`
e `max_new_tokens`; campos desconhecidos são rejeitados. Não há um campo
fictício de velocidade. Todos são opcionais: omita-os (ou envie `{}`) para
usar a configuração nativa do checkpoint Qwen, o baseline recomendado para a
primeira avaliação da qualidade do clone.

```bash
curl \
  -X POST \
  http://10.1.1.47:8001/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Olá, Pedro. Está tudo funcionando normalmente.",
    "voice": "giseli-10s",
    "language": "pt-BR",
    "seed": 42
  }' \
  --output aegis.wav
```

Com uma chave configurada, acrescente `-H "X-API-Key: SUA_CHAVE"` sem gravá-la
em scripts ou logs.

A resposta é `Content-Type: audio/wav`, PCM 16-bit mono, no sample rate
retornado pelo modelo. Não há resample ou efeito de áudio desnecessário. Ela
inclui:

```text
X-Aegis-TTS-Model
X-Aegis-TTS-Voice
X-Aegis-TTS-Seed
X-Aegis-TTS-Generation-Seconds
X-Aegis-TTS-Audio-Duration-Seconds
X-Aegis-TTS-RTF
X-Aegis-TTS-Queue-Wait-Seconds
X-Aegis-TTS-Sample-Rate
X-Aegis-TTS-Dtype
X-Aegis-TTS-VRAM-Before-MiB
X-Aegis-TTS-VRAM-After-MiB
X-Aegis-TTS-VRAM-Peak-MiB
```

O RTF é `tempo de geração / duração do áudio`. A aplicação mede VRAM com as
estatísticas do PyTorch antes, depois e no pico da inferência. O uso global da
GPU pode ser conferido com `nvidia-smi`.

Erros de contrato relevantes:

| HTTP | Situação |
| --- | --- |
| `401` | Chave ausente/incorreta quando autenticação está habilitada. |
| `422` | Payload, ID de voz, idioma ou parâmetros inválidos. |
| `503` | Serviço ainda não pronto, modelo/referência indisponível ou OOM recuperável. |
| `504` | Timeout de fila ou de geração reportado sem cancelar CUDA de modo inseguro. |

O serviço usa uma fila e locks para manter exatamente uma inferência na RTX
A2000. Se o cliente HTTP desistir, a chamada CUDA iniciada continua protegendo
a GPU até terminar de fato. Um OOM limpa cache quando seguro, retorna `503` e
mantém o processo vivo quando o modelo continuar utilizável.

### Contrato para futuro `TtsProvider` .NET

| Item | Valor |
| --- | --- |
| Base URL | `http://10.1.1.47:8001` |
| Método | `POST /v1/synthesize` |
| Autenticação | `X-API-Key` somente se configurada no Aegis TTS |
| Timeout cliente | Maior que fila + geração configuradas; trate `504` separadamente |
| Entrada | JSON com `text`, `voice`, `language`, `seed` opcional e allowlist de parâmetros |
| Voz padrão | `giseli-34s` |
| Idioma | `pt-BR` ou `Portuguese` |
| Saída | `audio/wav` em bytes; preserve os headers `X-Aegis-TTS-*` para telemetria |
| Falhas | Trate `422`, `503` e `504` sem assumir que uma resposta parcial é áudio |

O backend principal da Aegis não é alterado por este projeto.

## Smoke test

Depois de modelo e duas referências válidas:

```bash
./scripts/smoke-test.sh
```

O script inicia/atualiza o serviço, espera `/health/ready`, testa cada voz de
forma sequencial, salva exatamente dois WAVs em `samples/smoke/`, exibe os
headers de métrica, mostra VRAM e imprime logs recentes. Ele usa estas frases:

```text
Olá, Pedro. O serviço de voz da Aegis está funcionando corretamente. Pelo que eu verifiquei, está tudo normal por enquanto.
A Aegis integra PostgreSQL, Qdrant, Docker, Entity Framework e serviços OpenAI.
```

Caso as referências ainda não estejam neste servidor, o script falha cedo com
o comando de importação necessário; não inventa WAVs nem transcrições.

## Testes automatizados

Os testes comuns usam mocks e não baixam modelo. Para instalar as dependências
de desenvolvimento em um ambiente local separado:

```bash
# Ubuntu/Debian, se o módulo venv ainda não estiver instalado:
sudo apt install python3.12-venv

python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

`requirements-dev.txt` fixa `pytest`, `pytest-asyncio` e `httpx`, mas não é
instalado na imagem de runtime. Sem alterar o Python do host, a alternativa
efêmera abaixo instala apenas esses três extras no container descartável (as
dependências de runtime já estão na imagem):

```bash
docker run --rm \
  -v "$PWD:/workspace:ro" \
  -w /workspace \
  --entrypoint /bin/sh \
  aegis-tts:local \
  -lc 'python3.12 -m pip install --no-cache-dir --user \
       pytest==8.3.5 pytest-asyncio==0.25.3 httpx==0.28.1 && \
       PATH="$HOME/.local/bin:$PATH" PYTHONPATH=/workspace pytest -q'
```

O teste real de GPU é opcional e requer os pesos já presentes; execute-o
somente quando o marcador/variável descrito nos testes estiver habilitado.

## Operação, disco e logs

```bash
docker compose logs -f aegis-tts
docker compose ps
curl -s http://127.0.0.1:8001/health | jq .
nvidia-smi
docker image ls aegis-tts:local
du -sh models/Qwen3-TTS-12Hz-1.7B-Base cache samples
df -h .
```

Por padrão, todas as sínteses são retornadas em memória e não ocupam disco.
Com `AEGIS_TTS_SAVE_SAMPLES=true`, o serviço salva WAV e JSON em `samples/` e
retém no máximo `AEGIS_TTS_MAX_SAMPLES` itens, removendo apenas os samples mais
antigos. Ele nunca remove referências, modelos, caches externos ou recursos do
Docker globalmente.

Ao reportar uma instalação, registre a saída dos comandos acima para tamanho
da imagem, tamanho dos pesos, espaço restante, tempo de load, tempo de criação
dos prompts, VRAM pós-load, RTF por voz e resultado do smoke test. Esses
valores dependem da GPU, dos pesos e das referências locais e não devem ser
inventados no README.

## Remoção consciente

Para parar apenas este serviço, use `docker compose down` dentro deste
diretório. Bind mounts `models/`, `cache/`, `voices/` e `samples/` permanecem
intactos. Antes de apagar qualquer dado local, inspecione-o e remova somente o
subdiretório deste projeto que você realmente quer descartar. Não use
`docker system prune` para administrar este serviço.

## Limitações intencionais

- qualidade/identidade dependem da fidelidade da referência e de sua
  transcrição exata;
- não há garantia de controle preciso de velocidade, sotaque ou estilo;
- não há streaming, WebSocket, STT, uploads pela UI ou conversão automática;
- referências inválidas não são corrigidas pelo serviço;
- o modelo precisa ser baixado uma vez antes do startup e deve caber na GPU.
