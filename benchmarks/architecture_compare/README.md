# Comparação de arquiteturas — preparação isolada

Este cliente não carrega modelos, não baixa pesos, não treina, não altera serviços
e não inicia servidores. A operação padrão é um plano local, sem rede e sem GPU:

```bash
python3 benchmarks/architecture_compare/benchmark.py
```

O cliente `run` mede um runtime que o operador já preparou numa janela autorizada.
Somente HTTP em `127.0.0.1` ou `::1`, porta **18000 ou maior**, é aceito. O serviço
de produção não pode compartilhar a GPU: os PIDs CUDA observados devem coincidir
exatamente com `allowed_compute_pids`. Nenhum comando deste pacote interrompe a
produção. A liberação/restauração da GPU continua sendo uma operação separada,
aprovada pelo proprietário, com a reserva necessária para restaurar o serviço.

## Protocolo e configuração

Copie `candidate.example.json` para um diretório privado fora de qualquer checkout.
Preencha a revisão imutável do runtime, hashes reais de **todos** os pesos (incluindo
codec/decoder), PIDs CUDA e taxa nativa confirmada. `runtime_verified`,
`response_cache_disabled` e `native_rate_confirmed` só passam a `true` depois das
verificações. Os hashes são declarações de proveniência: o cliente não lê nem
revalida automaticamente os arquivos de pesos. Inclua no JSON privado os hashes da
referência/transcrição, frontend, dtype, opções de chunking, comando/ambiente, seed
efetivo e a evidência da granularidade de streaming do código fixado.

O contrato aceito é POST com JSON, texto no campo top-level `text_field` e resposta
**PCM mono s16le sem cabeçalho**, `Content-Type: audio/pcm` ou
`application/octet-stream`. O `payload` permite campos específicos do runtime,
incluindo referências locais previamente autorizadas. O texto do corpus substitui
apenas o campo configurado. `seed_field: null` declara que a seed não é controlada;
não se finge reprodução estocástica exata. Não há adaptação automática de SSE,
WAV, PCM float32 ou `audio/L16` (big-endian). Se o servidor retornar esses formatos,
prepare um adaptador explícito e registre sua revisão e custo.

O exemplo é um **contrato ainda não verificado**, não uma receita de inicialização
de Qwen, GPT-SoVITS ou VoxCPM2. Não se pressupõe suporte de GPT-SoVITS a vLLM.
O frontend/checkpoint PT-BR precisa existir e ser fixado antes de medir GPT-SoVITS.

Uma janela privada baseada em `window.example.json` registra a autorização real
do proprietário, candidata exata, início e fim UTC. O arquivo exemplo é inválido
para execução. Não transforme uma proposta de janela em aprovação.

```bash
python3 benchmarks/architecture_compare/benchmark.py plan \
  --config /CAMINHO_PRIVADO/candidate.json

# Apenas dentro da janela aprovada, com runtime isolado pronto:
python3 benchmarks/architecture_compare/benchmark.py run \
  --config /CAMINHO_PRIVADO/candidate.json \
  --window /CAMINHO_PRIVADO/window.json \
  --output /CAMINHO_PRIVADO/qwen06-run-01 --repetitions 3
```

`httpx` é necessário somente no modo `run`; os modos `plan` e `ab` usam a biblioteca
padrão. Não há instalação automática. Resultados/WAVs têm modo 0600, diretório
0700, fora de qualquer checkout; arquivos existentes não são sobrescritos.

## O que é medido

Cada observação inclui headers HTTP, primeiro byte, primeiro PCM completo, primeiro
PCM útil, EOF, duração e RTF (`tempo POST→EOF / duração do áudio`). O relógio começa
imediatamente antes do POST, depois de construir o cliente. O EOF é registrado ao
terminar a leitura, antes de fechar cliente/conexão. A primeira execução permanece
nos resultados; não há warm-up descartado. Modelo já carregado **não** representa
cold start de processo/modelo.

O PCM útil é um **proxy digital**, duas janelas completas consecutivas de 10 ms com
pico >100 e RMS >30 na escala PCM16. O tempo registrado corresponde à chegada dos
samples necessários para confirmar o onset, separado do offset no áudio. O
simulador respeita a chegada de cada bloco e samples divididos entre transportes;
conta underruns maiores que 10 ms. Isso não comprova fala inteligível, pronúncia
ou experiência num dispositivo real. Um clique isolado não passa como voz útil.

Mais de um chunk HTTP e entrega antes de EOF **não comprovam** streaming incremental
do modelo. A granularidade real (tokens/codec, vocoder ou frases completas) deve
ser confirmada no runtime e preservada no campo `streaming_runtime_evidence`.

Durante a janela, `nvidia-smi` amostra uso total e por PID aproximadamente a cada
100 ms, além do snapshot inicial. Esse valor inclui runtime/cache residente. Não
é o pico dos allocators PyTorch, pode perder picos curtos e não mede o carregamento
anterior ao benchmark. Registre separadamente pico de carga e alocações/reservas
quando o runtime oferecer métricas. A A2000 12 GB deste host expõe cerca de 11514
MiB: esse teto visível prevalece sobre uma conta abstrata de 12288 MiB.

Falhas são preservadas em JSON e interrompem a sequência. Não há seleção das
melhores gerações. O corpus inclui frases inéditas, nomes, nasalização, negação,
números crus, forma por extenso, siglas, datas e um trecho longo. Use o mesmo
corpus/ordem/repetições para todas as candidatas e marque normalização externa,
referência e DSP. Compare raw→raw para arquitetura e efeito final→efeito final
para produto; mudanças simultâneas não isolam causa. Execute uma segunda rodada
de maior duração somente se a rodada inicial justificar o custo.

O comando offline `python3 benchmarks/architecture_compare/benchmark.py summarize --results /PRIVADO/run/results.json`
imprime JSON com contagens, taxa de falhas e p50/p95/máximo de PCM útil, EOF e RTF
por candidata/caso, além de agregados explicitamente marcados como mistura de
casos. Não usa rede/GPU. Todos os resultados entram nos denominadores; as métricas
usam somente sucessos, sem excluir outliers. P50 é a mediana e p95 usa nearest
rank (`ceil(0,95*n)`); com poucas repetições, p95 frequentemente coincide com o
máximo e tem pouca precisão para a cauda. Para salvar, redirecione stdout para
um arquivo no diretório privado com `umask 077` aplicado pelo operador.

## A/B privado, inteiramente offline

Prepare um JSON privado com pares já gerados:

```json
{"pairs":[{"case":"novel_short","text":"A gaveta azul contém três chaves e um bilhete de Joana.","a":"/PRIVADO/qwen.wav","b":"/PRIVADO/candidata.wav","category":"architecture","same_generation":false}]}
```

```bash
python3 benchmarks/architecture_compare/benchmark.py ab \
  --pairs /CAMINHO_PRIVADO/pairs.json \
  --output /CAMINHO_PRIVADO/listening-01
```

Abra `index.html` localmente; não é necessário publicar servidor. A/B são
embaralhados de forma reproduzível; `answer-key.private.json` fica separado da
interface e não deve ser aberto pelo avaliador antes das notas. `review.json`
contém os critérios. O texto compartilhado é declarado pelo manifesto; o programa
não executa ASR para prová-lo. Categoria e `same_generation` distinguem controles
de DSP/reconstrução de comparações entre arquiteturas.

Os WAVs são copiados bit a bit, com taxas nativas preservadas e sem ganho,
recorte ou reamostragem. Registre diferença de loudness e escute também no mesmo
volume percebido por controle manual; nunca confunda áudio mais alto com maior
fidelidade. Notas de identidade, efeitos, bandas, ataques, sibilantes, números e
siglas continuam dependentes da escuta do proprietário.

## Runtime VoxCPM2 e controles Qwen

`voxcpm_runtime.py` é um adaptador nativo preparado, ainda sem smoke com pesos.
O plano padrão não importa Torch e não consulta GPU:

```bash
python3 benchmarks/architecture_compare/voxcpm_runtime.py
python3 benchmarks/architecture_compare/benchmark.py plan \
  --config benchmarks/architecture_compare/candidate.voxcpm2.unverified.json
python3 benchmarks/architecture_compare/benchmark.py plan \
  --config benchmarks/architecture_compare/candidate.gptsovits.unverified.json
```

O [relatório Vox/Qwen](../../docs/v2.0/voxcpm-qwen-evidence.md) contém o comando
de inicialização, pins e protocolo de reconstrução. O servidor só carrega pesos
locais após conferir janela, GPU exclusiva, fonte limpa e hashes. Desliga acesso
Hub e denoiser. `--optimize` está bloqueado devido ao fallback silencioso do
upstream com `cuda:0`; a primeira configuração é eager. O payload admite seed
uint32 por request. O modo preparado é clonagem por continuação com transcrição.
PCM16 é serializado sem normalização de ganho; float32 opcional fica em `.npy`.
Use `--save-float` em rodada de qualidade separada; o I/O ocorre depois do EOF
e pode atrasar a aceitação do pedido seguinte. Checks de GPU dentro do servidor
entram na TTFA HTTP e têm tempo separado no receipt nativo. O término do gerador
não prova que todo o texto foi falado, pois `max_len` pode encerrar a geração.

`import_qwen_history.py` recalcula métricas públicas históricas e, opcionalmente,
copia controles privados existentes sem modelo, rede, resampling ou nova síntese:

```bash
python3 benchmarks/architecture_compare/import_qwen_history.py \
  --summary-output /var/tmp/qwen-history-summary.json

# Destino privado novo fora de qualquer Git; nunca sobrescreve WAVs:
python3 benchmarks/architecture_compare/import_qwen_history.py \
  --summary-output /var/tmp/qwen-history-summary.json \
  --existing-captures /var/tmp/aegis-tts-latency-20261101 \
  --private-output /var/tmp/NOVO_CONTROLE_PRIVADO/qwen-controls
```

Os controles são Qwen antes/depois da revisão de latência e bruto→DSP.
**Não** são uma comparação GPT-SoVITS/VoxCPM2. A decisão corrente e os
impedimentos estão na [comparação de arquiteturas](../../docs/v2.0/architecture-comparison.md).
