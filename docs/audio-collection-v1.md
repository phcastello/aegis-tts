# AegisVoice v2: coleção real PT-BR

Esta receita aproveita os clipes tratados pelo proprietário. Não gera outro
corpus, não reaplica WORLD/W04/P07/M02 e não altera `AegisVoicev1.0`.
O importador legado continua exigindo seus próprios masters e aprovações.

## Dataset recebido

O ZIP entregue foi movido para armazenamento privado fora do checkout, extraído
com verificação CRC e conferido contra os hashes declarados no JSONL.
Foram recebidos 1.130 WAVs PCM24, mono, 48.000 Hz, totalizando
13.001,473208 s (3h36min41,47s), em 226 grupos e 36 categorias.
As durações medidas variam de 1,66056 a 15,06917 s.

Todos os 1.130 WAVs passaram pela importação técnica: nenhum hash divergente,
arquivo vazio/truncado, clipping, ID duplicado, grupo entre partições ou
variante textual próxima entre partições foi detectado. Isso não comprova
correspondência perceptiva entre áudio e texto nem fidelidade do efeito.
O ASR e a revisão do proprietário continuam sendo verificações distintas.

O relatório Windows identifica `schema=aegis.audio_collection.v1`; os registros
individuais não precisam repetir esse marcador. A seleção do adaptador é
explícita, por `--input-schema`, sem relaxar o importador legado.

Campos canônicos: `id`, `audio_filepath`, `text`, `split`, `group_id`, `category`,
`duration`, `sample_rate`, `channels`, `audio_sha256`. Metadados acústicos e de
recorte originais permanecem em `source_record`. Os aliases aceitos estão em
`training/audio_collection.py`; valores contraditórios são recusados.
Os companions reais `train.jsonl`, `validation.jsonl` e `test.jsonl` têm somente
caminho/texto/duração. Sua consistência é verificada contra o manifesto principal;
eles não fornecem nem substituem IDs, grupos ou decisões de split.

O pacote não contém masters nem manifestos originais de montagem. Sua
proveniência permanece `unverified`, sem hashes inventados. Quando esses arquivos
estiverem disponíveis, `--provenance` aceita:

```json
{"artifacts": [{"role": "manifest", "path": "originais/manifest.json"},
               {"role": "processed_master", "path": "originais/master_processed.wav"}]}
```

Os papéis permitidos são `manifest`, `source_master` e `processed_master`.
`sha256` esperado é opcional: sem ele o hash é computado, sem afirmar uma
verificação contra declaração prévia. Arquivos ausentes continuam não verificados.
A existência dos arquivos e seus hashes, por si só, não prova o vínculo temporal.

## Conversão e snapshot

O leitor verifica limites RIFF, chunks, formato PCM, frames, byte rate e block
align. Suporta PCM16 e PCM24, incluindo extensible PCM com largura válida igual
à do contêiner. PCM24 é decodificado com extensão de sinal; não é reinterpretado
como PCM16. Caminhos absolutos, escapes e symlinks fora da raiz são recusados.

O resampling usa `scipy.signal.resample_poly`, janela Kaiser beta 10, para mono
22.050 Hz. A quantização PCM16 recusa overshoot e arredondamento que causaria
wraparound. Não há normalização de loudness, limiter, mudança de velocidade,
trim ou aplicação de efeitos. Os originais permanecem intactos.

O snapshot é criado em staging e publicado uma única vez. Reexecuções exigem
outro destino. WAVs e manifestos preparados ficam somente leitura; os hashes
vinculam os registros, o CSV e as partições. Rejeições e motivos são inventariados
no relatório; falhas de integridade bloqueiam o runner.

O `metadata.csv` Windows de três colunas não é usado. O novo CSV single-speaker
tem duas colunas `arquivo.wav|texto`, com escaping CSV compatível com o
`csv.reader` do Piper. Contém somente treino e validação; o teste final não entra
no trainer, na reconstrução selecionada nem no ajuste de hiperparâmetros.

```bash
python3 -m training.prepare_dataset \
  --input-schema aegis.audio_collection.v1 \
  --manifest /PRIVADO/received/dataset_final/dataset.jsonl \
  --dataset-root /PRIVADO/received/dataset_final \
  --split-plan /PRIVADO/split-review-proposal.json \
  --output /PRIVADO/prepared/collection-v3-balanced
```

O plano é opcional. Sem ele, os splits originais são preservados. Um plano
explícito precisa identificar o hash fonte, todos os IDs, o hash canônico das
atribuições e grupos íntegros; não pode mover teste nem validação originais.
`source_split` e `source_record.split` preservam a atribuição anterior.

## Decisão de split

Cada grupo contém cinco falas de uma categoria. Portanto, 55 exemplos de
validação só podem representar 11 categorias mantendo os grupos íntegros.
Para ampliar sua representatividade, foram movidos 25 grupos completos de treino
para validação: um por categoria ausente, escolhido pelo menor SHA-256 de
`seed + NUL + group_id`, seed `40733133`. As escolhas usam somente metadados,
antes de medir o modelo. Os 50 exemplos de teste e a validação original ficaram
preservados.

| Partição | Exemplos | Grupos | Categorias | Duração preparada |
| --- | ---: | ---: | ---: | ---: |
| Treino | 900 | 180 | 36 | 10.345,606 s |
| Validação | 180 | 36 | 36 | 2.153,358 s |
| Teste final | 50 | 10 | 8 | 502,535 s |

A limitação de cobertura do teste final permanece explícita. Não foi corrigida
usando suas frases para escolher hiperparâmetros. O snapshot original
1.025/55/50 também foi preservado privadamente para rastreabilidade.

Hashes SHA-256 do experimento:

- Manifesto recebido: `c0cc5ac19fd833c5dc6bc416386aa2df9380296d38e4ec11d0046141c8269444`.
- Atribuições finais: `65fa79ef3c4990b7ffa158142518179b67b3a2045fc8ff6adc1908758c199380`.
- Dataset preparado: `935347d6bbe1db6058e22bee85ab53d4e89bbff9565d6fc998863a1daa6ee532`.
- Trainval preparado: `c45908e96f98556aed66b3c72718f8414923ea2576394b017833c293398fd8f0`.

## Qualidade, autorização e reconstrução

A declaração posterior do proprietário, “tudo aprovado.”, confirmou direitos,
revisão auditiva geral e janela operacional. O registro privado identifica essa
declaração e seu escopo, sem inventar provedor, voz, URL/data de termos ou lista
de IDs ouvidos. O importador continua produzindo `content_review=pending`;
a aprovação global fica em documento externo vinculado por hashes.

`training.collection_review` gera plano acústico, amostra por categoria/split,
extremos de duração/nível/espectro/cauda e casos com números/nomes/perguntas.
Suspeitas novas são preservadas para revisão específica. Aprovação global
prévia não é convertida em confirmação de uma reconstrução ainda não ouvida.

O ASR usa faster-whisper 1.1.1 com modelo local small, idioma `pt`, sem prompt
com a transcrição esperada e sem VAD. Recebe PCM validado convertido para
float32 mono a 16 kHz, evitando dependência da API de decodificação PyAV.
Mantém texto esperado e reconhecido, edições de palavras, segmentos, confiança,
hashes e progresso retomável. Não corrige textos automaticamente.
CPU int8 é o padrão. CUDA float16 é opt-in e exige autorização, janela exclusiva
e pelo menos 9.000 MiB livres; não pode coexistir com a produção carregada.

```bash
python3 scripts/dataset_content_audit.py \
  --dataset /PRIVADO/prepared/collection-v3-balanced/dataset.jsonl \
  --model-dir /PRIVADO/models/faster-whisper-small \
  --output-dir /PRIVADO/audit/asr --cpu-threads 4
python3 -m training.collection_review \
  --prepared /PRIVADO/prepared/collection-v3-balanced \
  --content-audit /PRIVADO/audit/asr/content-audit.jsonl \
  --audit-provenance /PRIVADO/audit/asr/content-audit-provenance.json \
  --output /PRIVADO/audit/review-plan.json
```

O plano do v3 sugere 102 amostras estratificadas. A inspeção acústica sinalizou
124 exemplos: 123 têm mais de 1% de potência acima da nova Nyquist de 11.025 Hz;
três apresentam outliers individuais de nível, centroide ou palavras/segundo,
com sobreposição entre motivos. Não foram detectados sinais de bordas cortadas,
silêncio ou clipping. O risco de banda é sistemático e não comprova defeito
perceptivo em cada clipe. A potência acima de Nyquist varia até 2,39%, com mediana
de 0,69%. Isso precisa ser confrontado com a textura desejada.

A base continua sendo Piper VITS/HiFi-GAN Cadu medium PT-BR; não houve troca de
arquitetura. O [card oficial](https://huggingface.co/rhasspy/piper-voices/blob/b4e358ffcb232591bf73a540e70f7de528e4174a/pt/pt_BR/cadu/medium/MODEL_CARD)
e a configuração confirmam 22.050 Hz, um locutor e frontend `pt-br`.
O checkpoint local tem 845.888.150 bytes e SHA-256
`abd3e6c7237aca5b7cf9c3cfc50945bfef44a83a594e00186f584d9d11fd355a`.
Todos os pesos do gerador, incluindo posterior/decoder, passaram pela conferência
de chaves e shapes. A identidade masculina da base é diferente da voz-alvo;
compatibilidade técnica não demonstra que o piloto aprenderá o timbre desejado.

Sete reconstruções reais em CPU, com 66,74 s de alvo no total, cobrem durações
de 1,66–15,07 s. RTF medido de 0,327–0,464, uma thread, sem clipping;
MAE log-espectral de 0,736–0,854. Um caso mudou RMS em −3,28 dB e centroide em
+501 Hz. Os pares privados A/B e seus hashes registram esse risco. O decoder
continua treinável; o resultado inicial não precisa ser equivalente, mas essas
métricas não aprovam identidade, robotização ou reverb.

## Ambiente, benchmark e piloto

Python 3.12 e dependências de `training/requirements.lock` foram instalados em
venv privado com `--require-hashes`, Torch 2.5.1+cu121. Nenhuma dependência foi
instalada no container de produção. O alinhador Cython ausente no wheel foi
compilado a partir da revisão Piper fixada; headers Python foram extraídos de
um pacote de desenvolvimento no diretório privado, sem instalação no host.
O receipt registra fontes, compilador, hashes e teste monotônico CPU.

O Piper 1.8 acrescentou cinco fonemas ingleses ao mapa padrão. O pipeline fixa
o mapa exato de 161 entradas da base usando `phonemes_path`, sem remapear seus
IDs. Verifica cada sequência fonética do cache contra o frontend aprovado.
Os 1.080 textos de treino/validação passaram pela cobertura `pt-br`, sem símbolos
desconhecidos. Isso não equivale a aprovação de pronúncia de números ou nomes.

A acumulação foi corrigida: o Piper usa otimização manual de G/D, incompatível
com `Trainer(accumulate_grad_batches=4)`. O runner acumula explicitamente os
gradientes dos dois otimizadores, separa seus parâmetros, trata o grupo parcial
final e avança os schedulers por época. Testes de autograd verificam os updates.

A prova CPU do data loader carregou todos os 1.080 exemplos e percorreu os
450/90 batches de treino/validação. O cache frio levou 42,67 s; preparação e
verificação completa, 59,49 s, com pico de RSS de 704,70 MiB. O maior batch de
treino tem áudio `[2, 1, 330093]` e espectrograma `[2, 513, 1289]`.
O teste final não foi carregado nessa prova.

Receita inicial: seed 40733133, batch 2, acumulação 4, FP32, taxas G/D
5e-5/2,5e-5, um worker. Com a partição escolhida, são 450 minibatches de treino,
90 de validação e 113 atualizações por otimizador por época; dez épocas somam
4.500 minibatches de treino. A partição original teria 512 por época.
Dez épocas são uma primeira avaliação, sem promessa de convergência.

O benchmark opt-in aquece oito minibatches e mede 32, incluindo um batch com os
clipes de treino mais longos. Sincroniza CUDA antes/depois de cada minibatch,
mede forward/backward/updates, média/mediana/máximo, validação inteira e pico de
VRAM. Projeta 10/30 épocas e atualiza a projeção após a primeira época completa.
Não guarda pesos ajustados do benchmark nem promove candidata.

```bash
python3 -m training.train_piper \
  --prepared /PRIVADO/prepared/collection-v3-balanced \
  --checkpoint /PRIVADO/models/pt_BR-cadu-medium.ckpt \
  --frontend-report /PRIVADO/audit/frontend-v3.json \
  --review /PRIVADO/audit/benchmark-review-v3.json \
  --output /PRIVADO/experiments/benchmark-fp32-v2 \
  --benchmark --execute \
  --owner-authorization REFERENCIA_REAL \
  --exclusive-gpu-window JANELA_REAL
```

Sem `--execute`, o comando somente imprime o plano. O piloto exige revisão de
fase `pilot`; o documento de benchmark não o autoriza. O histórico CSV registra
losses G/D/validação; JSONL registra tempos, recursos e falhas. NaN/Inf abortam;
divergência finita é sinalizada. São retidos os dois melhores checkpoints por
`val_mel` e o último, com índice SHA-256. `--resume` exige checkpoint do mesmo
experimento e identidade de dados/configuração/revisão preservada.

## Operação e rollback

A RTX A2000 tem 11.514 MiB visíveis. A produção ocupava 10.676 MiB, deixando
599 MiB livres: nenhum modelo de treino foi carregado junto dela.
O host dispõe de 12 CPUs lógicas (Xeon E5-2470 v2), cerca de 16 GiB de RAM
disponível na inspeção inicial. O proprietário liberou espaço; a preparação
prosseguiu após confirmar 33 GiB livres. Não foram removidos modelos/caches
para isso pelo pipeline.

Antes da janela, foram guardados privadamente inspect completo, imagem por ID,
compose, configuração, mounts e hashes dos 81 arquivos protegidos.
O container aprovado é `aegis-tts`, imagem `aegis-tts-gateway:latency-1749f5f`,
ID de imagem `sha256:8aef74c67fe5238ce1e2b8781b9da3aedc73784014dd38bc1c1553f741fe93a0`.
A operação retém esse mesmo container e o reinicia ao encerrar a janela, inclusive
quando o processo experimental falha. Não usa `compose down -v` nem recria volumes.
A retomada verifica readiness e síntese WAV real mono/PCM16/24.000 Hz.

Não há integração automática da candidata nos endpoints. Continuam preservados
`/v1/audio/speech`, `/v1/aegis/speech`, streaming, cancelamento, readiness e fila.
O adaptador candidato converte a taxa interna na borda e só poderá ser ligado
depois de aprovação perceptiva e testes reais de compatibilidade/latência.
`AegisVoicev1.0-Q00-W04-P07-M02` permanece o rollback aprovado.

Todos os dados, áudios A/B, modelos, configurações privadas e experimentos ficam
sob `/home/phc/aegis-private/aegisvoice-v2/`, fora do Git. O diretório é privado
e persistente. ZIPs e checkpoints também foram acrescentados aos ignores.
As remoções locais pré-existentes de `dataset/` não fazem parte desta mudança;
o teste legado que lê `dataset/corpus.pt-BR.jsonl` depende desse arquivo ausente.
