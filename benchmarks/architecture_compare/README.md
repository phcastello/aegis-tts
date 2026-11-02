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

## Investigação F5-TTS PT-BR / Magpie v2607

O estado atual e os resultados reais estão em
[comparação](../../docs/v2.0/architecture-comparison.md),
[F5](../../docs/v2.0/f5-evidence.md) e
[Magpie](../../docs/v2.0/magpie-evidence.md). Esses documentos substituem a antiga
prioridade GPT-SoVITS e o critério eliminatório de 32 kHz. Piper continua encerrado.

`f5_probe.py` acrescenta somente um runner offline específico; reaproveita os
checks de janela/GPU e o proxy de PCM útil existentes. O plano padrão é seguro:

```bash
python3 benchmarks/architecture_compare/f5_probe.py
python3 benchmarks/architecture_compare/magpie_codec_probe.py
```

F5 aceita `--execute --action roundtrip|infer --device cpu|cuda`, arquivos locais
verificados e referência com transcrição. A fonte precisa coincidir com o pin.
CUDA exige janela aprovada e GPU exclusiva; CPU esconde CUDA. O runner nunca
interrompe produção. NFE `8 16 32` usa Euler, referência mel em cache e frase
inteira. `--warmup` é opcional e preserva áudio/tempo; nenhuma chamada é descartada.
O formato nativo salvo é FLOAT WAV, sem normalização/trim/ASR. Métricas PCM16 só
são calculadas quando a saída cabe na faixa sem clipping. O custo da referência
é separado da inferência. O suplemento Aegis fica em corpus separado, sem mudar
os nove textos originais.

[artifacts.f5.json](artifacts.f5.json) fixa a lista mínima de quatro downloads,
revisões, bytes e hashes. O checkpoint acústico histórico `.pt` não é necessário.
Instale o ambiente fora de produção; o freeze real está em
[requirements-f5-cu118.lock.txt](requirements-f5-cu118.lock.txt). A wheel cu118
foi selecionada para o driver 535; execução CUDA ainda exige smoke real. Exemplo
para recriar o ambiente, com espaço previamente conferido:

```bash
python3 -m venv /PRIVADO/env-f5
/PRIVADO/env-f5/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
  -r benchmarks/architecture_compare/requirements-f5-cu118.lock.txt
```

O F5 roda pelo `source-dir` fixado e não precisa instalar Gradio, bitsandbytes,
ASR ou o conjunto completo de extras upstream. O freeze inclui dependências
indiretas exigidas pelo import do Trainer oficial e o auditor ASR CPU opcional
instalado nesta investigação; o F5 não chama esse auditor. Caches devem permanecer sob
uma raiz experimental privada e entrar no inventário de 40 GB; hardlinks uv
precisam ser contados uma vez no uso físico.

O probe NanoCodec é somente CPU e executa definições numéricas oficiais
verificadas, sem instalar NeMo completo. O probe de frontend confere fontes e
G2P/dicionário. Nenhum deles prova o runtime de síntese Magpie:

```bash
python3 benchmarks/architecture_compare/magpie_frontend_probe.py \
  --artifact-dir /PRIVADO/magpie
CUDA_VISIBLE_DEVICES='' /PRIVADO/env-f5/bin/python \
  benchmarks/architecture_compare/magpie_codec_probe.py --execute \
  --artifact-dir /PRIVADO/magpie --audio /PRIVADO/tratado.wav \
  --output /PRIVADO/NOVO_ROUNDTRIP
```

O roundtrip precisa partir do áudio tratado, preservando A original, B convertido
e C reconstruído. Se não houver original de 48 kHz, registrar sua ausência e a
taxa real, sem inventá-lo por upsampling. BigVGAN exige mel/checkpoint coerentes;
não trocar o decoder do checkpoint Vocos por simples coincidência de shapes.

## ZIP privado para ouvir em outro ambiente

O script package_listening.py reutiliza o empacotador A/B existente e lê somente
os WAVs/recibos explícitos desta investigação. Não acessa rede, GPU, pesos ou
ambientes. Precisa de NumPy/SoundFile para conversão explícita FLOAT→PCM16.

    CUDA_VISIBLE_DEVICES='' /PRIVADO/env-f5/bin/python \
      benchmarks/architecture_compare/package_listening.py \
      --experiment-root /var/tmp/aegis-v2-evaluation-20261102 \
      --output-dir /PRIVADO/NOVO_PACOTE \
      --zip /PRIVADO/NOVO_PACOTE.zip --require-nfe 8 16 32

O destino e o ZIP devem ser novos, fora do Git. A opção --require-nfe impede
entregar uma rodada final antes de todos os passos pedidos terminarem. Sem ela,
o pacote contém todas as saídas concluídas, sem escolher a melhor amostra.
Inclui os seis controles A↔B/B↔C/A↔C, a coorte Qwen de nove textos quando seu
manifesto está presente, comparações entre NFEs da mesma repetição e Qwen↔F5
somente com texto igual. Produção com DSP e F5 nativo são identificados na chave;
os tempos CPU e GPU não são comparáveis.

Depois de extrair **todo** o ZIP, index.html abre a escuta offline. O formulário
tem critérios por amostra, preferência por par e notas; os botões baixam JSON
ou CSV preenchido, sem servidor. Existe também ficha-vazia.csv. A coorte Qwen
identificada está em página separada, recomendada após a rodada cega. Áudios
numéricos originais, conversões, hashes, métricas, ASR limitado quando disponível
e identidades ficam em NAO_ABRIR_ANTES_DA_ESCUTA/. O ASR só entra se o hash
corresponder a uma amostra empacotada; não substitui a audição.

O ZIP exclui pesos, caches, credenciais, transcrições de referências adicionais
e caminhos absolutos do host. Conversões não aplicam ganho, corte, limiter,
denoiser ou resampling; erro de PCM16 limitado a um LSB. Se uma saída FLOAT
exceder a faixa, sua conversão é bloqueada e o original FLOAT permanece no
pacote e no par, com limitação de reprodução registrada. Não se corta o pico
nem se exclui a tentativa; suporte FLOAT e saturação na reprodução dependem
do navegador/dispositivo.

    python3 -m pytest -q tests/test_package_listening.py

Os testes verificam preservação de overload, remoção de caminhos privados,
inclusão de todas as tentativas e exclusão de arquivos não selecionados.
O ZIP é produzido localmente e permanece privado; o comando não o publica.
