# ASR auxiliar local — 02/11/2026

Foram transcritos 12 áudios: nove casos da coorte fixa Qwen atual e a frase
`novel_short` do F5 nos NFE 8, 16 e 32. O ASR recuperou o texto completo das três
amostras F5; isso auxilia a inspeção de omissões/repetições nessa frase, sem
comprovar pronúncia, identidade, textura robótica ou preferência perceptiva.
NFE32 foi transcrito numa segunda chamada local com as mesmas opções.

| Áudio | Resultado auxiliar; conferir pelo ouvido |
| --- | --- |
| F5 NFE 8, `novel_short` | Texto completo; diferença de capitalização. |
| F5 NFE 16, `novel_short` | Texto completo; diferença de capitalização. |
| F5 NFE 32, `novel_short` | Texto completo; diferença de capitalização. |
| Qwen, `novel_short` | Texto completo; diferença de capitalização. |
| Qwen, `novel_question` | Texto completo. |
| Qwen, `numbers_raw` | Moeda reconhecida como `AirHofAE83`; demais valores reconhecidos. Alerta para escuta. |
| Qwen, `numbers_spoken` | Conteúdo monetário e percentuais reconhecidos; ASR escreveu os valores com algarismos. |
| Qwen, `dates_units` | Data/hora/unidades reconhecidas; `mantenha` → `mantém a` no ASR. |
| Qwen, `acronyms` | Divergências em CPU, SUS e URL, incluindo repetição de `US`; conferir áudio e frontend. |
| Qwen, `br_phonetics` | `João em São Luís` → `João e São Luís`; nomes e locais restantes reconhecidos. |
| Qwen, `negation_tail` | Palavras recuperadas; pontuação final alterada pelo ASR. |
| Qwen, `long_stability` | `Se houver` → `Se eu ver`; restante recuperado, sem repetição detectada pelo ASR. |

Os alertas de moeda e siglas são do baseline corrente, não evidência de
superioridade do F5: somente uma frase F5 foi sintetizada até esta rodada.
A auditoria independente do gateway também sinalizou a normalização da moeda
literal; nenhuma configuração ou fonte da produção foi corrigida neste ensaio.
O ASR pode produzir falsos positivos, especialmente em siglas, nomes como Aegis,
valores e pontuação. Não houve score, seleção das melhores amostras, correção
automática das transcrições nem aprovação de conteúdo.

Foi usado [SYSTRAN/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120),
revisão `536b0662742c02347bc0e980a01041f333bce120`, licença MIT declarada no card.
Baixaram-se somente `model.bin`, `config.json`, `tokenizer.json` e
`vocabulary.txt`: **486.212.372 bytes**, em aproximadamente 16,38 s.
O `model.bin` é CTranslate2, não pickle PyTorch; seu SHA-256 é
`3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671`.
Os tamanhos/hashes públicos de todos os componentes estão no
[manifesto sanitizado](asr-artifacts.json). O blob grande foi validado contra
SHA-256 LFS do Hub; os arquivos pequenos, contra os IDs Git blob declarados,
com SHA-256 local também registrado.

Ambiente privado F5 já existente, acrescido de `faster-whisper==1.1.1`,
`ctranslate2==4.6.0`, `onnxruntime==1.22.1`, `av==14.2.0`,
`coloredlogs==15.0.1`, `flatbuffers==25.12.19` e `humanfriendly==10.0`.
Torch `2.6.0+cu118` e NumPy `1.26.4` permaneceram iguais. Uma tentativa de
preparação com AV 14.4.0 falhou por falta de wheel Linux/Python 3.12 e bibliotecas
FFmpeg de build; nenhum pacote foi instalado nessa tentativa. O pin 14.2.0 foi
instalado por wheel, com `--only-binary :all:`.

A execução usou CPU int8, duas threads, um worker, nice 19 e
`CUDA_VISIBLE_DEVICES=''`. Idioma `pt`, beam 5, temperatura 0,
`vad_filter=False`, `condition_on_previous_text=False`, `initial_prompt=None` e
`local_files_only=True`. Nenhum áudio foi enviado a serviço externo.
O WAV foi lido em float32 mono e reamostrado para 16 kHz por `resample_poly`
Kaiser beta 10, sem ganho, trim ou filtros de limpeza. Foram preservados hashes
dos WAVs e da entrada reamostrada do ASR.

A carga CPU levou 1,37 s; a rodada completa levou 144,33 s, dos quais 142,93 s
nas transcrições de 96,18 s de áudio. Estes são custos do auditor CPU, não
latência TTS. A segunda chamada adicionou NFE32 em 11,846 s de transcrição;
foram **12 tentativas e 12 conclusões** no total, sem descarte.
A duração foi limitada a aproximadamente cinco minutos pelo script.

Recibos, transcrições completas e o comando reproduzível ficam em
`/var/tmp/aegis-v2-evaluation-20261102/asr/`, com arquivos privados 0600.
`run-limited.py` confere os hashes das observações já feitas e processa somente
novos áudios existentes; foi assim que NFE32 entrou na segunda chamada:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  nice -n 19 /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  /var/tmp/aegis-v2-evaluation-20261102/asr/run-limited.py
```

Antes de reproduzir, conferir `download-receipt.json`, `dependencies.json` e
os hashes do manifesto fixado. A leitura dos recibos não exige carregar modelos.
O script não altera o TTS nem concede autorização de treino.
