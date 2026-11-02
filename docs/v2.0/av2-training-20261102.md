# AegisVoice v2 — execução de fine-tuning em 02/11/2026

Estado deste registro: preparação CPU, piloto de três épocas e vinte e duas épocas de
treinamento concluídos; treinamento completo em andamento após a retomada de
uma falha recuperável no hook de avaliação.
Avaliações CUDA finais e restauração definitiva ainda estão pendentes.
O proprietário autorizou a interrupção temporária do Qwen para **todas** as
operações necessárias na GPU, sem prazo cronológico predefinido. A nova voz não
será implantada automaticamente. Sua aprovação definitiva depende da escuta do
proprietário.

## Dataset definitivo

O ZIP `dataset_final_124_7ppm.zip` substitui o corpus anterior. Seu SHA-256 completo
foi validado como
`5bbd8288d9ab795a83ffd1599b0c5a32c00b401cb903b368a183d7ee7580c8df`.
São 1.871.602.459 bytes comprimidos, 2.274 entradas e 2.063.227.270 bytes
extraídos. Todas as entradas passaram na verificação CRC e na validação de
caminhos; não foram aceitos traversal, caminhos absolutos, links, dispositivos,
destinos duplicados ou arquivos criptografados.

| Partição fixa | Falas | Duração preparada em segundos |
| --- | ---: | ---: |
| Treino | 1.025 | 13.030,095583 |
| Validação | 55 | 722,993542 |
| Teste final | 50 | 546,257000 |
| Total | 1.130 | 14.299,346125 |

Os originais têm **14.299,334042 segundos**, aproximadamente 3h58min19s, mono,
48 kHz, PCM24. Os 12,083 ms adicionais somados nos arquivos preparados decorrem
do arredondamento para amostras inteiras ao reamostrar cada trecho para 24 kHz.
Não há exemplo rejeitado ou em quarentena, sample em clipping ou token OOV na
auditoria realizada. Os IDs de WAVs e textos, os três índices de partição e as
duas colunas textuais do CSV coincidem exatamente com o manifesto. Grupos,
áudio idêntico e texto normalizado não cruzam partições. Isso verifica a
consistência documental; não equivale à escuta ou ao alinhamento independente
palavra a palavra de todas as falas.

A conversão usa `scipy.signal.resample_poly`, janela Kaiser beta 10, mantendo
PCM24. Não aplica ganho, efeitos, denoise, pitch, velocidade ou corte. O mel é
o `MelSpec(vocos)` da revisão F5 fixada, calculado em CPU a partir do WAV de
24 kHz efetivamente escrito: 100 canais, FFT/janela 1.024, hop 256, magnitude,
escala HTK e log com piso `1e-5`. Os caches são `.npy` float32 `[100,T]` sem
pickle. Os hashes de cada original, WAV convertido e mel foram reconferidos.

O texto de treino recebe NFC, remoção dos espaços externos e minúsculas
explicitamente, preservando `original_text`. O frontend é
`convert_char_to_pinyin` upstream; não há phonemização PT-BR inventada ou
substituição silenciosa do vocabulário. `raw.arrow` no nível superior contém
somente treino; cada partição também tem seu próprio Arrow, durações e
vocabulário em `prepared/splits/`.

O proprietário aprovou perceptivamente o corpus tratado: menos expressivo,
pitch ligeiramente reduzido e textura feminina sintética/robótica. Relata ter
recebido autorização por e-mail da ElevenLabs para o uso pessoal dos outputs
transformados no treinamento da Aegis. Esta declaração está registrada no
recibo privado, sem verificação independente do e-mail e sem extensão a uso
comercial ou a outras finalidades. Permanece necessária a guarda privada da
resposta original e dos cabeçalhos; isso não cria uma nova aprovação pendente
para a execução já autorizada.

## Arquitetura e versões

Usa-se `F5TTS_Base` PT-BR do firstpixel, com Vocos congelado, aproveitando o
ambiente de `feat/v2-f5-magpie-evaluation`. A arquitetura tem 337.096.804
parâmetros: largura 1.024, 22 blocos, 16 heads, FF2, texto 512, quatro camadas
convolucionais, `text_mask_padding=False` e `pe_attn_head=1`. A carga é estrita.
Não se troca pela configuração `F5TTS_v1_Base` apenas porque alguns shapes
coincidem. As verificações anteriores e as licenças constam em
[F5: evidência reproduzível](f5-evidence.md).

| Artefato | Revisão / SHA-256 |
| --- | --- |
| Código SWivid/F5-TTS | commit `45868abaabd74aabf03cd673d6dd00cc9d9c01ae` |
| firstpixel/F5-TTS-pt-br | revisão `0af67bc061699cab46218c150948910fde209614` |
| `model_last.safetensors` brasileiro | `c25e2678bdd4304071e3d0e39a283a7d23ffbc23678080c5b0b741dda2e07f0f` |
| Vocabulário, 2.545 entradas | `2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c` |
| Vocos | revisão `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21` |
| `pytorch_model.bin` Vocos | `97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a` |
| `config.yaml` Vocos | `da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc` |

O ambiente preparado permanece isolado em
`/var/tmp/aegis-v2-evaluation-20261102/envs/f5/`: PyTorch/torchaudio
2.6.0+cu118, NumPy 1.26.4, datasets 2.21.0 e pyarrow 17.0.0. O lock completo
é [requirements-f5-cu118.lock.txt](../../benchmarks/architecture_compare/requirements-f5-cu118.lock.txt).
A RTX A2000 apresenta 11.514 MiB utilizáveis; os 12 GB comerciais não são
tratados como 12 GiB livres. O Qwen 0.6B/vLLM-Omni continua sendo o baseline e
o rollback. Piper e GPT-SoVITS não integram esta execução.

## Preparação e configuração inicial de treinamento

A preparação privada ocupa 1.576.001.536 bytes e a extração 2.070.208.512 bytes,
fora o ZIP. O orçamento adicional é de 40 GB, contando também os ambientes e
artefatos experimentais já presentes; o espaço real livre e a margem de
rollback devem prevalecer. Checkpoints usam retenção limitada: melhor, piloto,
intermediário e último, mais um estado completo retomável substituído de forma
atômica. Não há retenção por época ilimitada.

O engine usa diretamente a loss CFM da fonte fixada. Os defaults iniciais são
microbatch de uma fala completa, acumulação oito, BF16 autocast, parâmetros e
momentos AdamW FP32, activation checkpointing, learning rate `1e-5`, warmup de
100 updates, weight decay `0.01`, clipping de gradiente em `1.0`, seed
`20261102` e dois threads CPU. Vocos permanece congelado e não há EMA residente
na GPU. Os defaults são uma hipótese inicial; os recibos registram os valores
realmente executados e cada ajuste motivado por evidência.

Antes do treino são medidos updates reais com forward, loss, backward,
acumulação, optimizer step e validação, incluindo falas curtas, medianas e
longas. O runner qualifica acumulação quatro e oito. A projeção de duração é
informativa. Não existe deadline de treinamento, nem parada simplesmente ao
atingir dez épocas.

A validação usa as 55 falas fixas com os mesmos estados de máscara, ruído,
tempo e dropout por item em cada avaliação. O piloto é promovido
automaticamente após pelo menos três épocas quando há melhoria de validação
superior a 0,5%. A falta dessa melhoria até seis épocas encerra o piloto. Na
fase completa, depois de ao menos 12 épocas, seis avaliações sem melhoria
relativa de 0,5% reduzem o learning rate uma vez pelo fator 0,3. Um novo platô
encerra por convergência. Deterioração superior a 10% em relação ao melhor
valor por três avaliações também pode encerrar o treino. NaN/Inf, gradientes
não finitos, falta de espaço ou falhas CUDA são preservados como falhas técnicas,
sem fabricar conclusão de treinamento. O teste final não decide esses ajustes.

## Progresso confirmado e recuperação

O piloto completou três épocas e **387 updates**, com promoção automática para
a fase completa. A loss fixa de validação caiu de **0,784124** no checkpoint-base
para **0,597315**, melhoria de aproximadamente **23,8%** no objetivo CFM.

| Época | Updates acumulados | Loss média de treino | Loss de validação |
| --- | ---: | ---: | ---: |
| 1 | 129 | 0,767497 | 0,619051 |
| 2 | 258 | 0,686593 | 0,603426 |
| 3 | 387 | 0,667024 | 0,597315 |
| 4 | 516 | 0,669986 | 0,593350 |
| 5 | 645 | 0,657427 | 0,590924 |
| 6 | 774 | 0,655744 | 0,587778 |
| 7 | 903 | 0,651552 | 0,586404 |
| 8 | 1.032 | 0,654430 | 0,584650 |
| 9 | 1.161 | 0,654888 | 0,583988 |
| 10 | 1.290 | 0,661020 | 0,582228 |
| 11 | 1.419 | 0,647378 | 0,580772 |
| 12 | 1.548 | 0,650571 | 0,579771 |
| 16 | 2.064 | 0,640273 | 0,576533 |
| 22 | 2.838 | 0,659611 | 0,573859 |

A tabela mostra as épocas iniciais e marcos posteriores; `loss-history.json`
e a curva privada conservam todas as avaliações. Na época **22**, foram
concluídos **2.838 updates**, com validação **0,573859398**, melhoria de
**26,82%** em relação à base. O learning rate recebeu a **única redução
prevista, por fator 0,3**, após seis avaliações sem superar o ganho relativo
de 0,5% desde a última melhoria relevante. A melhor loss absoluta ainda vinha
caindo lentamente; isso não equivale a uma deterioração generalizada.

A paciência foi reiniciada e a fase completa prossegue na época 23, observando
a resposta à redução. O treinamento **ainda não terminou**: um novo platô
persistente após essa redução poderá encerrar por convergência conforme a
regra inicial. O valor baixo do learning rate, o número de épocas ou os
indicadores auxiliares isolados não determinam o encerramento.

Ao término da época doze, foram concluídos **1.548 updates**, com validação
**0,579770833**, melhoria de **26,06%** em relação à base. O checkpoint
intermediário dessa época foi preservado. A fase completa prosseguiu na época
treze: a paciência estava em uma avaliação, sem redução do LR por platô. A
época doze é o mínimo inicial para aplicar critérios de convergência, sem
constituir uma meta de encerramento. Nenhuma busca extra de hiperparâmetros
foi iniciada com base nesses dois áudios.

Na sexta época, foram concluídos 774 updates, com melhoria de 25,04%.
No hook NFE 16 da época seis, as disponibilidades de PCM completo foram
1,859 s para `novel_short` e 1,976 s para `aegis_command`, com RTF 0,454 e
0,491. São chamadas no processo de treino, com os estados do treinamento
residentes; não incluem HTTP e não substituem os benchmarks finais isolados.

Esses tempos registram a disponibilidade do buffer completo, antes da reprodução.
Os proxies digitais encontram 430 ms e 390 ms de áudio antes do início da
atividade, respectivamente. Com reprodução FIFO imediata, o primeiro início
estimado seria **2,289 s** e **2,366 s**, excluindo dispositivo e cliente. O proxy
F5 exige dois frames consecutivos de 10 ms com pico PCM16 superior a 100 e RMS
superior a 30. O TTFA operacional Qwen mede a chegada do primeiro bloco HTTP
com algum sample absoluto superior a 32; não é o instante de reprodução audível.
Os métodos diferentes impedem declarar substituição direta ou vantagem de
latência a partir desses números. O relatório mantém separadamente
`ttfa_ms`, `first_pcm_ms`, `onset_audio_offset_ms` e
`estimated_first_audible_playback_ms`, conservando os recibos individuais.

Na sexta época, o ASR CPU `faster-whisper-small`, int8, quatro threads, idioma português,
beam cinco, temperatura zero e sem VAD transcreveu `novel_short` com a
sequência “e um”, mas registrou `Joana` como `juana`: WER de 9,09%. No comando,
registrou `Agis, confir no registro...`, WER de 33,33%. Esses dois áudios não
são suficientes para concluir a qualidade da voz, a pronúncia de Aegis ou o
comportamento de textos longos. A candidata continua sem aprovação auditiva.

No checkpoint doze, os mesmos textos foram sintetizados com o prompt 0586,
NFE 16, seed `40733133` e repetição zero, mantendo a comparação com as épocas
três e seis. `novel_short` foi transcrito integralmente pelo ASR, WER de 0%; o
comando preservou toda a sequência, com `Aegis` reconhecido como `Agis`, WER
de 11,11%. A divergência `confir no` da época seis não apareceu nesse áudio.

| Frase fixa | WER época 3 | WER época 6 | WER época 12 | WER época 22 |
| --- | ---: | ---: | ---: | ---: |
| `novel_short` | 18,18% | 9,09% | 0,00% | 0,00% |
| `aegis_command` | 11,11% | 33,33% | 11,11% | 11,11% |

Na época 22, a reexecução ASR canônica de quatro threads confirmou novamente
essas duas transcrições: a primeira frase coincide integralmente; o comando
continua com `Aegis` reconhecido como `Agis`. Foram reutilizados 15 registros
anteriores e computados somente os dois novos. O resultado não resolve a
pronúncia desejada do nome, que permanece para escuta do proprietário.

O recibo suplementar da época doze usa o mesmo modelo/decodificação e dois
threads CPU; a reexecução canônica com quatro threads confirmou WER de 0% e
11,11%, reutilizando o cache das observações anteriores. Os recibos e
configurações permanecem separados no relatório, sem agrupar WER de políticas
diferentes. Essa comparação de duas frases não aprova a
voz nem caracteriza comportamento em frases longas. A redução de WER não é
usada isoladamente como critério de parada ou mudança de hiperparâmetros.

No hook da época doze, o buffer completo ficou disponível em 1,917 s e
1,792 s, com RTF 0,468 e 0,446, respectivamente. Os inícios estimados de
reprodução FIFO são 2,337 s e 2,182 s. Continuam medições no processo de treino,
sem HTTP, com a mesma distinção de métodos descrita acima.

A comparação ASR usa somente NFD, remoção de acentos, casefold e separação
de pontuação/espaços. Não aplica o léxico ou o frontend da produção. Os
parâmetros, o hash de configuração e o recibo dos pesos estão em
`asr/settings.json`; o relatório agregado preserva também hashes dos scripts
na coleta do relatório. Transcrições ASR e WER permanecem indicadores auxiliares.

Às 11:52:15 UTC, o hook de síntese da terceira época falhou com
`ModuleNotFoundError: No module named 'benchmarks.architecture_compare'`. Um
pacote instalado chamado `benchmarks` ocultava os helpers locais. O checkpoint
retomável já havia sido salvo no início da época quatro, update 387; os pesos,
estado do otimizador e histórico foram preservados. A correção usa a carga
explícita do helper local e passou na verificação CPU com o ambiente F5 real.

O guardião executou seu rollback após o encerramento do subprocesso. Essa foi
uma restauração intermediária indesejada: a primeira janela começou a parada
às **11:08:45,992 UTC** e concluiu a restauração verificada às
**11:55:17,327 UTC**, totalizando **46min31,334s** de indisponibilidade.
O mesmo container foi novamente inspecionado e testado saudável, e a segunda
janela começou às **11:57:06,433 UTC**, com recibos próprios em
`operations-resume-01/`. O engine retomou automaticamente a época quatro,
update 387, às 11:57:57 UTC; a fase completa continua, sem novo limite de tempo.

Os recibos das duas janelas permanecem preservados. O relatório agrega as
indisponibilidades efetivas, excluindo o intervalo em que a produção esteve
restaurada. A restauração da primeira janela não é apresentada como estado final
enquanto a janela retomada estiver ativa. A melhoria de loss ainda não comprova
pronúncia, identidade robótica ou aprovação perceptiva da candidata.

## Acompanhamento acústico em CPU

`training.f5_acoustics` acompanha `novel_short` e `aegis_command`, mantendo
NFE 16, repetição zero e a seed registrada em cada recibo. Reutiliza RMS,
centroide espectral, flatness e proporções de energia por banda existentes no
projeto. Conserva as métricas na taxa nativa e também analisa todos os áudios
em 24 kHz, com janela 1.024/hop 256, para controlar diferenças de banda e janela
entre originais de 48 kHz e saídas de 24 kHz. Os WAVs permanecem intactos.

A F0 é estimada por `librosa.yin`, faixa de 60 a 500 Hz. A máscara aceita frames
com RMS acima do maior valor entre -55 dBFS e 30 dB abaixo do pico de RMS do
arquivo. Essa máscara mede atividade energética; não é um classificador de
vozeamento. Fricativas, efeitos sintéticos e erros de oitava podem produzir
candidatos incorretos. São descritores auxiliares, sem critério de parada
independente ou demonstração de identidade vocal.

Na primeira coleta, o prompt `aegis_ptbr_0586` apresentou mediana YIN de
196,19 Hz tanto no original como no convertido. Na época três, as medianas
foram 209,94 Hz para `novel_short` e 210,74 Hz para `aegis_command`. Os textos
são diferentes do prompt; essas diferenças não isolam um efeito do treinamento.
As comparações longitudinais posteriores mantêm texto, NFE, repetição e seed.
Na época seis, essas mesmas frases apresentaram medianas YIN de 208,98 Hz e
208,40 Hz, respectivamente: diferenças de -0,96 Hz e -2,33 Hz em relação à
época três. São mudanças descritivas das amostras, sem comprovação de qualidade
ou fidelidade da identidade.
Na época doze, as medianas das mesmas frases foram 208,90 Hz e 208,55 Hz,
respectivamente. Os gráficos foram atualizados com as três observações por
frase, mantendo os descritores como auxiliares da escuta.
Na época 22, foram 209,18 Hz e 209,46 Hz. O arquivo acústico agora acompanha
quatro checkpoints representativos, sem atribuir aprovação vocal às diferenças.

Foi planejada uma única comparação final adicional: no melhor checkpoint,
NFE 16 e duas frases fixas, derivar a duração da saída neural da média ponderada
de **124,60845 palavras/minuto** do treino. O benchmark principal continua com
a regra upstream baseada em bytes UTF-8 e NFE 8/16/32. O plano está em
`tempo-comparison-plan.json` e não modifica waveform, velocidade por efeito,
pitch ou demais efeitos. Palavras de comprimentos diferentes e pontuação
alteram naturalmente a contagem por minuto; a variante não pressupõe erro de
ritmo no treinamento nem inicia uma busca de combinações.

Nos três roundtrips Vocos já disponíveis, a mediana YIN mudou de -3,93 a +0,55 Hz,
o RMS de -0,42 a -0,05 dB e o centroide de -1,22 a +28,45 Hz na análise comum
de 24 kHz. Isso registra alterações acústicas, sem substituir a escuta dos
efeitos robóticos. A reconstrução Vocos do prompt 0586 não estava disponível;
não foi gerada uma nova reconstrução para esta medição.

Os resultados privados estão em `acoustics/results.json`, com hashes dos WAVs
e dos caches de análise, além dos gráficos independentes `acoustics/f0.png` e
`acoustics/spectrum.png`. O gerador do relatório incorpora esses resultados e
mantém a necessidade de avaliação perceptiva do proprietário.

## Operação e baseline anterior à interrupção

O snapshot preserva o container original
`08004c66f014e3e63fb6fd123946e4dbba2919ce29b4530848279917325faa75`, imagem
`sha256:8aef74c67fe5238ce1e2b8781b9da3aedc73784014dd38bc1c1553f741fe93a0`,
configuração, mounts e hashes das referências protegidas. O rollback inicia o
mesmo container. Nenhum volume, imagem, cache ou checkpoint é apagado.

Em 02/11/2026 às 11:02 UTC, o Qwen estava saudável. Uma síntese WAV levou
2,803 s. Na síntese PCM real, o primeiro bloco recebido com algum sample de
amplitude absoluta superior a 32 chegou em 1,662 s; total de 2,685 s para
3,730 s de áudio, RTF 0,720. A GPU registrava 10.674 MiB usados. É uma observação
operacional de uma frase, sem percentis ou controle de carga fria.

Cada janela GPU usa um processo guardião para treino e avaliações CUDA. A
restauração ocorre no `finally` depois do término desse processo; um watchdog
sem prazo reage à perda do guardião. A política da janela retomada mantém o
Qwen desligado até a conclusão de todas as operações CUDA. A restauração
intermediária provocada pela falha anterior está registrada acima. Após a liberação da GPU, o rollback
confere identidade/configuração/referências e testa readiness, health, síntese
WAV e PCM streaming. CPU ASR, curvas e empacotamento podem terminar com o Qwen
já restaurado quando não demandam CUDA.

## Reprodução e artefatos privados

Raiz privada: `/var/tmp/aegisvoice-v2-training-20261102/`. WAVs, transcrições,
estados do otimizador, recibos operacionais completos e respostas de e-mail
não devem ser adicionados ao Git.

```bash
# Preparação CPU com Qwen ativo.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.f5_prepare \
  --zip /home/phc/aegis-tts/dataset_final_124_7ppm.zip \
  --root /var/tmp/aegisvoice-v2-training-20261102 \
  --f5-source /var/tmp/aegis-v2-evaluation-20261102/f5/source \
  --vocab /var/tmp/aegis-v2-evaluation-20261102/f5/vocab.hf.txt

CUDA_VISIBLE_DEVICES='' /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.f5_engine --action check \
  --run-root /var/tmp/aegisvoice-v2-training-20261102

CUDA_VISIBLE_DEVICES='' /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.f5_evaluate --action roundtrip \
  --run-root /var/tmp/aegisvoice-v2-training-20261102

# Janela única; requer snapshot e verificações CPU concluídos.
# Em reprodução usar uma nova raiz de operação, nunca sobrescrever o snapshot.
python3 -m training.av2_ops \
  --directory /var/tmp/aegisvoice-v2-training-20261102/operations \
  snapshot
python3 -m training.av2_ops \
  --directory /var/tmp/aegisvoice-v2-training-20261102/operations \
  verify
python3 -m training.av2_ops \
  --directory /var/tmp/aegisvoice-v2-training-20261102/operations \
  run -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.f5_run --run-root /var/tmp/aegisvoice-v2-training-20261102

# Após todas as operações CUDA e a restauração: ASR, pacote offline e curvas.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.f5_acoustics \
  --run-root /var/tmp/aegisvoice-v2-training-20261102
CUDA_VISIBLE_DEVICES='' /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.f5_evaluate --action asr \
  --run-root /var/tmp/aegisvoice-v2-training-20261102
CUDA_VISIBLE_DEVICES='' /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.f5_evaluate --action package \
  --run-root /var/tmp/aegisvoice-v2-training-20261102
CUDA_VISIBLE_DEVICES='' /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.f5_report --run-root /var/tmp/aegisvoice-v2-training-20261102
```

Os recibos principais são `prepared/audit.json`, `cpu-check.json`,
`roundtrip/results.json`, `benchmark-*.json`, `events.jsonl`, `loss-history.json`,
`best.json`, `training-result.json`, `evaluations/checkpoint-manifest.json`,
`evaluations/block-streaming.json`, `asr/results.json`,
`operations/gpu-window.json`, `operations/restored.json` e `report/report.json`.
As curvas são `report/loss.png` e `report/training-performance.png`. O pacote
privado tem índice HTML offline e manifesto SHA-256 dos áudios. O relatório
agregado conserva ausências como pendências, sem inventar métricas.

Testes CPU desta preparação: dez casos do limite de extração em
`tests/test_f5_preparation.py` passaram, incluindo caminhos maliciosos, symlinks,
hash incorreto, destinos duplicados, CRC corrompido e permissões privadas.
Seis regressões adicionais em `tests/test_f5_report.py` passaram: soma das
indisponibilidades de duas janelas, escolha da restauração mais recente e
manutenção da pendência enquanto uma retomada está ativa; um manifesto acústico
também não pode satisfazer a pendência do pacote auditivo. A estimativa YIN foi
conferida com seno sintético de 150 Hz e silêncio (sem frames ativos).
O teste de latência confirma que disponibilidade do buffer e reprodução
estimada continuam separadas, sem alterar os recibos individuais.
Recibos opcionais do proxy Qwen e da reprodução no navegador permanecem
ausentes até serem criados, sem fabricar resultados.
As avaliações CUDA
finais, o tempo total de indisponibilidade, os checkpoints selecionados, as
métricas finais e os commits serão registrados após a execução.
