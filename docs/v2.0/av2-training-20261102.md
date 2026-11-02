# AegisVoice v2 — relatório final de fine-tuning em 02/11/2026

A rodada foi concluída: **29 épocas, 3.741 updates e todas as avaliações CUDA
finalizadas**. O Qwen original foi integralmente restaurado e está saudável.
A melhor candidata está preservada para avaliação auditiva; **a identidade,
os efeitos robóticos e a voz definitiva ainda dependem da escuta do proprietário**.
A candidata não foi implantada na Aegis.

## Dataset

Foi usado exclusivamente o novo `dataset_final_124_7ppm.zip`, substituindo o
corpus anterior. O SHA-256 integral conferido foi
`5bbd8288d9ab795a83ffd1599b0c5a32c00b401cb903b368a183d7ee7580c8df`.
O ZIP tem 1.871.602.459 bytes; suas 2.274 entradas totalizam 2.063.227.270 bytes
extraídos. Foram validados CRC, caminhos, duplicações e tipos de arquivo, com
rejeição de traversal, caminhos absolutos, links e dispositivos.

| Partição preservada | Falas | Duração preparada, segundos |
| --- | ---: | ---: |
| Treino | 1.025 | 13.030,095583 |
| Validação | 55 | 722,993542 |
| Teste final | 50 | 546,257000 |
| Total | 1.130 | 14.299,346125 |

Os originais são mono, 48 kHz, PCM24, com duração total de **3h58min19,334s**.
A preparação para 24 kHz PCM24 adiciona apenas 12,083 ms somados, pelo
arredondamento de cada trecho para amostras inteiras. Não houve fala rejeitada
ou em quarentena, clipping de origem ou token OOV. IDs, hashes de todos os
WAVs, transcrições `.txt`, CSV e índices foram conferidos. Grupos, áudio
idêntico e texto normalizado não cruzam partições. A consistência documental
não equivale à escuta palavra a palavra de todo o corpus.

A conversão usa `resample_poly`, Kaiser beta 10. Não reaplica efeitos, ganho,
normalização, pitch, velocidade, denoise ou cortes. O cache usa o
`MelSpec(vocos)` fixado: 100 mels, FFT/janela 1.024, hop 256, magnitude, HTK e
log com piso `1e-5`, em `.npy` float32. Todos os hashes dos originais,
convertidos e mels foram reconferidos. O texto recebe NFC, strip e minúsculas
explícitas; `original_text` permanece preservado. O frontend é o
`convert_char_to_pinyin` upstream. Os arquivos Arrow e as durações de treino,
validação e teste ficam separados.

O proprietário aprovou perceptivamente o dataset tratado, com voz feminina
sintética/robótica, menor expressividade e pitch ligeiramente reduzido. Relata
ter recebido autorização por e-mail da ElevenLabs para o uso pessoal dos
outputs transformados no treinamento da Aegis. A declaração foi registrada
sem verificação independente e sem extrapolação para uso comercial ou outras
finalidades. Permanece necessária a guarda privada da resposta original e dos
cabeçalhos; a execução já estava expressamente autorizada pelo proprietário.

## Arquitetura e configuração

Foi usado **firstpixel/F5-TTS-pt-br, F5TTS_Base, com Vocos congelado**. A carga
foi estrita, com 337.096.804 parâmetros: largura 1.024, 22 blocos, 16 heads,
FF2, texto 512, quatro camadas convolucionais, `text_mask_padding=False` e
`pe_attn_head=1`. Não foi usada a configuração semanticamente diferente v1.
Qwen 0.6B/vLLM-Omni foi preservado como baseline e rollback; Piper e
GPT-SoVITS não foram retomados.

| Artefato | Revisão ou SHA-256 |
| --- | --- |
| Código SWivid/F5-TTS | `45868abaabd74aabf03cd673d6dd00cc9d9c01ae` |
| firstpixel/F5-TTS-pt-br | `0af67bc061699cab46218c150948910fde209614` |
| Checkpoint-base | `c25e2678bdd4304071e3d0e39a283a7d23ffbc23678080c5b0b741dda2e07f0f` |
| Vocabulário, 2.545 entradas | `2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c` |
| Vocos | `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21` |
| Pesos Vocos | `97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a` |
| Configuração Vocos | `da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc` |

O ambiente isolado é `/var/tmp/aegis-v2-evaluation-20261102/envs/f5/`, com
PyTorch/torchaudio 2.6.0+cu118. O
[lock completo](../../benchmarks/architecture_compare/requirements-f5-cu118.lock.txt)
e [as evidências de arquitetura/licença](f5-evidence.md) permanecem versionados.
Os pesos brasileiros são CC BY-NC 4.0; as licenças permissivas do código e do
Vocos não retiram essa restrição.

Configuração executada: microbatch de uma fala completa, acumulação oito,
BF16 autocast, parâmetros e momentos AdamW FP32, activation checkpointing,
LR inicial `1e-5`, warmup de 100 updates, decay inverso à raiz do contador,
weight decay `0.01`, clipping de gradiente `1.0`, seed `20261102`, dois threads
CPU e sem EMA durante o ajuste. A validação fixa usa as 55 falas com
ruído, máscara, tempo e dropout reproduzíveis por item. O teste final não
participou da seleção nem das decisões de parada.

## Treinamento e seleção

O piloto de três épocas foi promovido automaticamente após aprendizagem
estável. O treino terminou às **17:33:13 UTC**, em **29 épocas**, **3.741 updates**
e **29.725 microbatches**. A validação caiu de **0,784123678** para
**0,572552035**, redução de **26,98%** no objetivo CFM.

| Marco | Updates | Loss média de treino | Loss de validação |
| --- | ---: | ---: | ---: |
| Base | 0 | — | 0,784124 |
| Época 3 | 387 | 0,667024 | 0,597315 |
| Época 6 | 774 | 0,655744 | 0,587778 |
| Época 12 | 1.548 | 0,650571 | 0,579771 |
| Época 22 | 2.838 | 0,659611 | 0,573859 |
| Época 29 | 3.741 | 0,654057 | 0,572552 |

O critério foi `validation_converged_after_plateau_and_single_lr_reduction`:
após seis avaliações sem melhoria relativa de 0,5%, o LR recebeu a única
redução prevista, por fator 0,3, na época 22. Uma melhoria relevante na época
23 reiniciou a paciência; as seis avaliações seguintes não atingiram novo
ganho relativo de 0,5%. Houve pequenos ganhos absolutos até o final, mas
insuficientes segundo o critério definido. Não houve limite de horas nem
encerramento por atingir dez épocas.

O tempo efetivo nos updates foi **20.374,979 s — 5h39min34,979s**, incluindo
forward, loss, backward, acumulação e optimizer step. Os dois subprocessos de
treino somaram **22.629,770 s — 6h17min09,770s**, incluindo validações,
checkpoints, cargas e hooks internos. Esses valores excluem o intervalo de
restauração/retomada e a avaliação CUDA final. A série completa e os gráficos
estão em `loss-history.json`, `report/loss.png` e
`report/training-performance.png`, incluindo a base na época zero.

O melhor checkpoint foi o da **época 29/update 3.741**, pela menor loss de
validação fixa. Foram preservados:

| Arquivo privado | Função / SHA-256 |
| --- | --- |
| `checkpoints/best.safetensors` | Melhor; `b3faf799babfcdf33768beb8dcc26f7f94c3c297b3b3a89549ab32cb9380fb44` |
| `checkpoints/pilot.safetensors` | Época 3; `2f9cccce5cd1f728251e5676b0397f688f0b166ecd951914aebd0e7fed2b59bc` |
| `checkpoints/intermediate.safetensors` | Época 12; `825e00412190c779a03ae2660ad10d40647bd34c5a911291ec3d5b8d2d7c5802` |
| `checkpoints/last.safetensors` | Último; `f4a36513df48efb26803bbd26e3771a7b147c1c0aa5a4d310f115805233edd37` |
| `checkpoints/last.pt` | Retomável; `c86af589edcedf2f0a58e174227b09e547ff7d60ced57beb23b18034f33e549f` |

O estado retomável contém modelo, otimizador, scaler, RNG e progresso. Sua
leitura CPU com `weights_only=True` passou em **389 verificações**, incluindo
estados finitos do otimizador e contador 3.741. A retenção foi limitada, sem
acumular um checkpoint completo por época.

O pico de treino foi **6.037,436 MiB alocados / 6.244 MiB reservados pelo
PyTorch**. Não houve OOM ou NaN/Inf reportado nos updates concluídos. Antes do
treino, benchmarks reais qualificaram acumulação quatro e oito, com medianas
de **2,792 s/update** e **5,459 s/update**, respectivamente, incluindo forward,
backward e optimizer; ambas também executaram validação. A projeção inicial
com acumulação oito foi aproximadamente 711,6 s/época, informativa e sem
funcionar como prazo de operação.

O problema recuperável foi um pacote instalado `benchmarks` ocultando os
helpers locais no hook da época três. O guardião restaurou o Qwen antes do
previsto. O conflito foi corrigido com carga explícita do helper local,
testado no ambiente F5, e o treino retomou a época quatro/update 387 com o
estado salvo. A restauração intermediária e a indisponibilidade das duas
janelas foram preservadas no relatório.

## Qualidade e limitações

As avaliações finais carregaram estritamente base, piloto, intermediário,
melhor e último checkpoint. Foram concluídas **117 sínteses primárias**:
base em dez casos × NFE 8/16/32; melhor em dez casos × três NFEs × duas seeds;
e piloto/intermediário/último em três casos × três NFEs. Também foram avaliados
blocos de sentenças e uma variante exploratória de duração. O pacote inclui
frases inéditas PT-BR, fonética, pergunta, negação, datas, números, siglas,
comando Aegis e texto longo, com comparações por frase e referências tratadas.

O teste final de 50 falas, usado somente após a seleção, caiu de **0,841037470**
na base para **0,667900618** no melhor checkpoint: redução de **20,59%**, com
loss menor nos 50 pares. Isso confirma melhoria do objetivo acústico nessa
partição, sem comprovar identidade vocal ou qualidade perceptiva.

O ASR canônico possui **151 registros**, todos com a mesma configuração:
`faster-whisper-small` fixado, CPU int8, quatro threads, português, beam cinco,
temperatura zero, sem VAD e sem condicionamento no texto anterior. O scorer
aplica somente NFD, remoção de acentos, casefold e separação de pontuação;
não usa normalização ou léxico do gateway. Recibos diagnósticos anteriores
mantêm seus próprios parâmetros, sem combinar WER de políticas diferentes.

Na comparação NFE 16/seed zero das quatro frases comuns mais curtas, o WER
agrupado por palavras passou de **9,62% para 3,85%**; a segunda seed do melhor
teve **7,69%**. São quatro frases/52 palavras, sem conclusão estatística ampla.
A melhora não foi uniforme em todos os casos e NFEs. O nome Aegis continuou
aparecendo como `Agis` em transcrições; números, siglas, nomes e a leitura
desejada ainda exigem escuta. Diferenças de formatação numérica também podem
inflar WER.

**Texto longo sem segmentação tem uma limitação relevante.** O ASR sinalizou
caudas alteradas/repetidas no melhor checkpoint em NFE 16/32 nas duas seeds,
e em parte dos resultados NFE 8. A presença de energia no final do waveform
não comprova as palavras efetivamente pronunciadas. Esses resultados não
permitem declarar fala longa plenamente estável. Os áudios e diagnósticos
foram preservados para escuta; ASR é um alerta auxiliar, não prova auditiva.

A alternativa por sentenças produziu **três de três blocos com WER zero**,
incluindo o final do terceiro. Os blocos ficaram disponíveis em
**1,975 / 3,954 / 6,215 s**, para durações de **6,987 / 7,125 / 8,864 s**.
O produtor permaneceu adiante da reprodução FIFO neste ensaio, sem underrun
estimado. Os saltos nas duas junções foram 0,000664 e 0,000531. Isso apoia
**segmentação por sentenças como requisito de integração**, com avaliação
perceptiva das junções. Cada bloco ainda exige ODE e Vocos completos: não
foi demonstrado streaming neural dentro do bloco nem servidor F5 via HTTP.

A única variante extra de duração, NFE 16 derivado de **124,60845 palavras/minuto**
do treino, teve resultado misto/negativo: `novel_short` apresentou WER de
**36,36%**, contra **9,09%** na regra principal nas duas seeds; o comando
manteve 11,11% e a divergência do nome. A variante permanece como A/B
exploratório e **não é a opção recomendada**. A regra upstream de duração e
os NFEs 8/16/32 do benchmark principal foram preservados. Não houve efeito de
velocidade, pitch ou processamento de waveform; o comprimento das palavras
e a pontuação impedem tratar WPM diferente como erro automático de ritmo.

Os três roundtrips original tratado → 24 kHz → Vocos tiveram transcrições ASR
sem divergências. Seus descritores mostraram mudanças pequenas de RMS e
pitch estimado, mas não estabelecem preservação dos efeitos robóticos.
`acoustics/results.json`, `f0.png` e `spectrum.png` acompanham F0 YIN, centroide,
flatness, bandas e RMS em análise comum de 24 kHz. YIN usa máscara de energia,
que não classifica vozeamento e pode errar em vozes sintéticas. Expressividade,
textura feminina/robótica, fidelidade à identidade e continuidade percebida
**não foram aprovadas por escuta humana nesta execução**.

Os WAVs FLOAT originais foram preservados, inclusive overshoot: o melhor
checkpoint apresentou 49 samples fora de `[-1,1]` em dois áudios NFE 8;
nenhum nos seus NFE 16/32. A base apresentou 1.271 samples fora da faixa em
26 de 30 áudios. Isso é **overshoot medido no FLOAT**, sem clipping,
normalização ou limiter aplicados para ocultá-lo. As cópias PCM16 para o
navegador foram recusadas quando a conversão exigiria alteração de ganho ou
clipping; os originais continuam disponíveis.

## Desempenho e integração

As sínteses usaram Euler, CFG `2.0`, sway `-1.0`, `use_epss=False` e
BF16 autocast com parâmetros FP32. A referência completa de condicionamento
foi `aegis_ptbr_0586`, com 10,103 s, pertencente ao treino. A matriz principal
manteve a duração upstream pela razão de bytes UTF-8; o controle por palavras
por minuto aparece em um grupo separado.

Medições locais do melhor checkpoint, dez casos e duas seeds por NFE:

| NFE | TTFA nativo / latência total mediana | RTF mediano | Pico PyTorch alocado |
| --- | ---: | ---: | ---: |
| 8 | 0,923 s | 0,148 | 2.154 MiB |
| 16 | 1,826 s | 0,291 | 2.164 MiB |
| 32 | 3,657 s | 0,584 | 2.183 MiB |

O TTFA nativo é a disponibilidade do buffer completo, incluindo frontend,
difusão e Vocos. Exclui HTTP e carga/preparo único do modelo e da referência.
A carga do melhor levou 4,504 s; na carga havia aproximadamente 1.362 MiB
alocados pelo PyTorch e 1.506 MiB totais no `nvidia-smi`. Para `novel_short`
NFE 16, as duas observações levaram 1,644 e 1,626 s para 4,096 s de áudio.
No texto longo de 23,147 s, levaram 4,107 e 4,128 s, com as ressalvas de
qualidade da cauda descritas acima.

O relatório separa `first_pcm_ms`, disponibilidade de PCM útil, offset de
atividade e reprodução audível estimada. O proxy exige dois frames de 10 ms,
pico PCM16 >100 e RMS >30, sob reprodução FIFO imediata, sem atraso de
cliente/dispositivo. O Qwen também teve o onset recalculado offline com esse
mesmo helper sobre o PCM e o cronograma real dos blocos, preservando o escopo
HTTP/DSP da produção.

Após o rollback, o Qwen apresentou TTFA HTTP operacional de **1,662 s**
(primeiro bloco com algum sample absoluto >32), total **2,679 s**, **3,730 s**
de áudio e **RTF 0,718**; a síntese WAV real levou **2,919 s**. A coorte fixa
histórica de nove requisições WAV Qwen também está preservada no pacote.
Os escopos, aquecimento e durações de áudio diferem; RTF local menor não
comprova ganho equivalente de TTFA em produção.

A candidata tem viabilidade técnica de memória e throughput na A2000, mas a
integração exige segmentação por sentenças, validação do frontend PT-BR,
controle de filas/entrega PCM e escuta de pronúncia, efeitos e junções.
Nenhum endpoint foi substituído. NFE 16 constitui um ponto de comparação para
a escuta; a escolha de voz e configuração definitiva cabe ao proprietário.

## Operação e recursos

Todas as operações CUDA terminaram em **02/11/2026 às 17:39:23,568 UTC**.
O F5 foi descarregado antes de iniciar o mesmo container Qwen, cuja restauração
com health/readiness, WAV real e PCM streaming foi concluída em
**17:42:09,512 UTC**. A produção foi novamente verificada saudável após a
finalização CPU. ID, imagem, configuração, mounts e 77 arquivos do bundle
protegido coincidem com o snapshot original:

- Container: `08004c66f014e3e63fb6fd123946e4dbba2919ce29b4530848279917325faa75`.
- Imagem: `sha256:8aef74c67fe5238ce1e2b8781b9da3aedc73784014dd38bc1c1553f741fe93a0`.
- VRAM residente observada: **10.674 MiB**, igual ao baseline operacional.

A indisponibilidade conservadora total foi **23.494,414 s — 6h31min34,414s**,
somando início do comando de parada até verificação de restauração em cada
janela. A primeira durou 46min31,334s e terminou no rollback recuperável de
11:55:17 UTC. A segunda começou às 11:57:06 UTC e terminou às 17:42:09 UTC.
O intervalo restaurado entre as duas foi excluído. Nenhum volume, imagem,
checkpoint, cache ou referência de produção foi apagado.

O inventário final, contando hardlinks uma única vez, registra
**24.778.014.720 bytes** experimentais, incluindo ambientes anteriores, execução
atual e ZIPs. O limite adicional de **40.000.000.000 bytes** foi respeitado;
restaram **26.146.496.512 bytes livres**, acima da reserva operacional de
**5 GiB**. O snapshot final é `report/resource-final.json`, posterior às
cópias PCM16 e ao ZIP. Diretórios privados têm modo 0700, arquivos 0600 e
scripts executáveis 0700. A nova voz não foi promovida à produção.

## Entrega privada e reprodução

Raiz dos resultados: `/var/tmp/aegisvoice-v2-training-20261102/`.
O pacote para escuta é
`/home/phc/aegisvoice-v2-finetuning-20261102.zip`, **132.077.331 bytes**, SHA-256:

`1011041b409dff753d133f5fc816bc923230bfdfb97e2266c7cd34e6915ad077`

O ZIP tem índice HTML offline, **163 WAVs nativos e 124 cópias PCM16**, com
**287 entradas WAV** no manifesto. Trinta conversões foram recusadas para
preservar a política sem ganho/clipping; os nativos FLOAT foram mantidos.
Todas as entradas passaram no CRC; os alvos locais do HTML e os hashes dos
WAVs nativos/cópias foram conferidos. Há áudio tratado, F5 antes do ajuste,
checkpoints treinados e Qwen para os casos cujo baseline já existia. As cópias
usam arredondamento para PCM16 sem efeitos, dither, resampling ou normalização.
O índice agrupa comparações pela mesma frase e identifica proveniência.

| Caminho na raiz privada | Conteúdo |
| --- | --- |
| `prepared/audit.json` | Auditoria integral do dataset e splits |
| `prepared/owner-authorization.private.json` | Declaração do proprietário e escopo relatado |
| `training-result.json`, `best.json`, `loss-history.json` | Treino, seleção e evolução |
| `checkpoints/` | Pesos e estado retomável |
| `evaluations/` | Sínteses, cargas, teste final e blocos |
| `asr/results.json`, `asr/settings.json` | 151 observações canônicas e configuração |
| `acoustics/` | Descritores e gráficos de F0/espectro |
| `listening/index.html` | Comparação offline e cópias de compatibilidade |
| `report/report.json` | Relatório final; única pendência: escuta do proprietário |
| `report/final-gpu-analysis.json`, `report/final-quality-analysis.json` | Síntese das evidências finais |
| `report/loss.png`, `report/training-performance.png` | Curvas de treino/validação e desempenho |
| `report/operation-final-audit.json`, `report/resource-final.json` | Rollback e recursos |
| `report/checkpoint-final-resume-audit.json` | 389 verificações do último estado retomável |
| `report/package-verification.json` | CRC, links locais e SHA-256 do pacote |
| `code-final/`, `code-provenance-final.json` | Cópias e hashes finais dos scripts |

A preparação é reproduzível com `training.f5_prepare`; o carregamento CPU com
`training.f5_engine --action check`. A execução CUDA usa
`training.av2_ops run` envolvendo `training.f5_run`, após snapshot e checks.
Esses comandos interrompem produção e exigem o contexto operacional apropriado;
esta rodada já foi concluída. A sequência CPU executada depois da restauração
está preservada em `run_cpu_finalize.sh` na raiz privada. Seus componentes são:

```bash
CUDA_VISIBLE_DEVICES='' /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.f5_evaluate --action asr --run-root /var/tmp/aegisvoice-v2-training-20261102
CUDA_VISIBLE_DEVICES='' /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.f5_acoustics --run-root /var/tmp/aegisvoice-v2-training-20261102
CUDA_VISIBLE_DEVICES='' /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.f5_evaluate --action package --run-root /var/tmp/aegisvoice-v2-training-20261102
```

O finalizador também executou o recálculo do proxy Qwen, a compatibilidade
PCM16, a verificação do ZIP/HTML e `training.f5_report` com hashes completos.
O script preservado contém os argumentos exatos e as asserções de GPU
finalizada e Qwen restaurado. Os recibos opcionais do proxy e da compatibilidade
estão presentes no relatório final, sem referências recursivas ao próprio JSON.

## Testes e commits

Foram aprovados **51 testes CPU, zero falhas**: 45 testes via pytest do sistema
e seis testes do engine no ambiente F5. Também passaram compilação de nove
módulos, `git diff --check`, auditorias de snapshot/rollback, cargas estritas,
minibatches CUDA reais, teste reservado, sínteses NFE 8/16/32, blocos FIFO,
verificação retomável e validação integral do pacote. Comandos registrados em
`report/cpu-tests-final.json`:

```bash
python3 -m pytest -q tests/test_av2_ops.py tests/test_f5_preparation.py \
  tests/test_f5_probe.py tests/test_f5_evaluation.py tests/test_f5_report.py \
  tests/test_f5_asr_metrics.py tests/test_f5_listening_compat.py
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m unittest tests/test_f5_engine.py
```

O intervalo de implementação é `01f16f0..8c40a38`, com Conventional Commits.
Entre os marcos: `453393e` (treino e guardião), `f245213` (autorização CUDA),
`d419b50` (correção de importação/retomada), `e81ea75` (ASR sem normalizador do
gateway), `912632d` (cópias PCM16), `de2b136` (frontend incluído na latência) e
`8c40a38` (registro do platô/redução de LR). Hashes e cópias finais de código
preservam a versão efetivamente entregue. Não há aprovação humana da nova voz
implícita nos testes, nas métricas ou no pacote.
