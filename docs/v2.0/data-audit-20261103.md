# Auditoria independente dos dados e F5 — 03/11/2026

Os **1.130 WAVs reais** do corpus final foram reabertos em CPU e seus hashes,
transcrições TXT, textos originais dos manifests preparados, WAVs preparados e
mels foram reconferidos. Nenhuma alteração foi aplicada aos insumos. A auditoria
de todos os tensores do melhor F5 e a carga estrita da arquitetura e do Vocos
também passaram em CPU. Isso verifica integridade e compatibilidade, sem alegar
escuta, identidade, correspondência fonética palavra a palavra ou inferência
CUDA nesta auditoria.

## Insumos recuperáveis

| Insumo privado | Caminho | SHA-256 |
| --- | --- | --- |
| ZIP original de entrada | `/home/phc/aegis-tts/dataset_final_124_7ppm.zip` | `5bbd8288d9ab795a83ffd1599b0c5a32c00b401cb903b368a183d7ee7580c8df` |
| Manifesto do corpus extraído | `/var/tmp/aegisvoice-v2-training-20261102/extracted/dataset_final_124_7ppm/dataset.jsonl` | `d36661e8876f7f3b48dfabfb7aad900987c16ad95021ea48b67a35e160cf800d` |
| Melhor F5 especializado, época 29/update 3741 | `/var/tmp/aegisvoice-v2-training-20261102/checkpoints/best.safetensors` | `b3faf799babfcdf33768beb8dcc26f7f94c3c297b3b3a89549ab32cb9380fb44` |
| Vocabulário compatível | `/var/tmp/aegis-v2-evaluation-20261102/f5/vocab.hf.txt` | `2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c` |
| Pesos Vocos | `/var/tmp/aegis-v2-evaluation-20261102/f5/vocos/pytorch_model.bin` | `97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a` |

A extração original ocupa cerca de 2,0 GiB; o preparo compartilhado cerca de
1,5 GiB. Não houve nova extração ou cópia de áudio. Os caminhos em `/srv`,
`/opt` e `/mnt` não forneceram outros modelos ou corpus. Os masters históricos
referidos pela procedência estavam no Windows do proprietário; os clipes
exportados e transformados são os insumos reais presentes. Os WAVs em
`synthetic-demo` são tons e estão excluídos.

O F5 especializado tem **337.096.804 parâmetros, 364 tensores e 1.348.432.240
bytes**; todos os tensores foram lidos em CPU e são finitos. Arquitetura
`F5TTS_Base` da revisão `45868abaabd74aabf03cd673d6dd00cc9d9c01ae`,
vocabulário de 2.545 entradas e Vocos carregaram com `strict=True`, todas as
chaves correspondentes. O recibo é
`/var/tmp/aegis-v2-work/data-audit/f5-strict-cpu-load.json`.
O ambiente já existente usa PyTorch `2.6.0+cu118`; não foi instalado outro.

## Formato, qualidade e separação

Todos os originais são **48.000 Hz, mono, PCM24**. A duração integral é
**14.299,334042 s (3h58min19,334s)**, com **zero amostras com clipping e zero
amostras próximas do clipping em |x|≥0,999**. Pico máximo 0,697898; pico mediano
0,480720. São arquivos já tratados com o caráter feminino sintético/robótico,
conforme aprovação histórica relatada pelo proprietário. Não se reaplicaram
efeitos, normalização ou resampling.

Diagnóstico de silêncio por RMS de frames de 10 ms, limiar −50 dBFS: **18,39%**
da duração em frames abaixo do limiar. Isso inclui pausas internas; não é VAD
de fala nem prova de duração útil. Silêncio inicial mediano 140 ms, p95 160 ms,
máximo 180 ms; final mediano 160,7 ms, p95 190,8 ms, máximo 369,7 ms. Não se
cortou áudio com esse diagnóstico.

| Partição preservada | Falas | Duração original (s) | Vocábulos únicos ausentes do treino |
| --- | ---: | ---: | ---: |
| Treino | 1.025 | 13.030,084750 | — |
| Validação | 55 | 722,992958 | 119 em 47 falas |
| Teste | 50 | 546,256333 | 90 em 39 falas |

A tokenização lexical desta contagem é NFC/casefold e palavras Unicode, sem
normalização de números; ela prova oportunidade de testar generalização
lexical, sem comprovar acerto de pronúncia. Não há duplicatas exatas de WAV,
nem texto canônico, ID de fala original ou `group_id` compartilhado entre
partições. Os grupos conservam variantes próximas na mesma partição.

Os **seis masters cruzam as partições**, mas a procedência os identifica como
montagens posteriores de falas individuais para edição no Melodyne/Audacity,
com separadores silenciosos, não como seis gravações contínuas originais.
Por isso o plano selecionado conserva a fala original por ID e os grupos
semânticos. A inspeção de índices não é uma comparação de PCM com masters
ausentes nem uma prova sobre sessões de geração não documentadas.

Um plano de sensibilidade opcional bloqueia os masters 1–4 para treino, 5 para
validação e 6 para teste; quarentena de dez falas cujos grupos cruzam essas
fronteiras mantém variantes juntas. Ele teria 810/180/130 falas e
10.674,225/2.583,329/898,007 s. Esse plano **não foi selecionado**: altera a
distribuição temática e não transforma seus holdouts em inéditos para o F5
histórico, que já viu parte deles.

As âncoras selecionadas são três falas integrais da validação original,
escolhidas deterministicamente por duração antes das saídas das candidatas.
O manifesto privado `identity-anchors.private.json` conserva caminhos,
transcrições e hashes. O teste final permanece separado das decisões de
treinamento. Contagem de metadados/lexemas não é avaliação do teste acústico.

## Reprodução e limites

```bash
cd /home/phc/aegis-v2-lab
CUDA_VISIBLE_DEVICES='' /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_data_audit \
  --source /var/tmp/aegisvoice-v2-training-20261102/extracted/dataset_final_124_7ppm \
  --prepared /var/tmp/aegisvoice-v2-training-20261102/prepared \
  --output /var/tmp/aegis-v2-work/data-audit \
  --archive /home/phc/aegis-tts/dataset_final_124_7ppm.zip \
  --checkpoint /var/tmp/aegisvoice-v2-training-20261102/checkpoints/best.safetensors
```

Os manifests reutilizáveis selecionados estão em
`/var/tmp/aegis-v2-work/data-audit/historical-preserved/{train,validation,test}.jsonl`.
Os manifests F5 existentes em
`/var/tmp/aegisvoice-v2-training-20261102/prepared/` oferecem tanto
`source_audio_filepath` (48 kHz) quanto `audio_filepath` (24 kHz) e mels Vocos;
um modelo com outro extrator mel deve produzir seu próprio cache a partir do
áudio, respeitando as mesmas partições.

`summary.json` contém métricas compactas; `records.private.json` registra cada
checagem e silêncio por fala. Nada privado é enviado ao Git. A integridade de
TXT/manifests/CSV já era documentada na preparação histórica; esta auditoria
reconfere TXT e manifesto preparado, mas não refaz escuta integral ou ASR do
corpus. Só a avaliação acústica comum e a escuta identificada podem decidir
pronúncia, timbre, sibilantes e manutenção do caráter robótico.
