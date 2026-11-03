# Cadu: baseline de idioma em CPU

Em 2026-11-03, o Cadu ONNX produziu os sete textos de validação nas variantes
`raw` e `refinement`: 14/14 WAVs completos, sem OOV, vazio, NaN ou overshoot.
O ASR CPU terminou os 14 casos. A carga e o frontend estão qualificados para
pesquisa local; a clareza ainda tem divergências que exigem diagnóstico, e a
identidade da Aegis não foi avaliada. Não houve audição, treino, upload, mudança
de perfil/gateway/serviço ou uso de GPU por esta execução.

O FastPitch077 continuou na GPU durante síntese e ASR. A inferência Cadu usou
somente `CPUExecutionProvider`, intra-op2/inter-op1, execução sequencial,
niceness19 e CUDA oculta. Estes tempos não são benchmark de produção nem
latência da A2000. A síntese terminou às09:58:26 UTC; o ASR executou de10:00:00
a10:03:05 UTC, sem outro decoder/ASR ativo na largada e antes do próximo marco
4000 do monitor de qualidade FastPitch.

## Artefatos e receita

Tudo permanece em `/var/tmp/aegis-v2-work/piper/`:

- `downloads/pt_BR-cadu-medium.onnx`: 62.950.044 bytes, SHA256
  `765f0809a6ea9035d4a6d0d008dbf8876e68b2dd32029312672fa8f405bdb535`.
  Origem `rhasspy/piper-voices@e7229f7d33a10e145e109b886d661984201ac3c4`.
- `deps/`: wheel `piper-tts1.8.0` abi3 extraído, sem instalar dependências ou
  alterar o ambiente F5/Torch. Wheel original34.131.442 bytes, SHA256
  `25b4d3f31ff70c8fa7151908e00aaa5650cbdf16bca8fcf21299f3941b89a7d3`.
  Após conferir o manifesto da extração e a execução real, a cópia compactada
  foi removida; origem/hash permanecem em `download-receipt.json`.
- `base-cadu-cpu/`: `recipe.json`, `frontend.json`, `overlay-manifest.json`,
  `results.json`, `done.json`, 14 FLOAT nativos22.050Hz, 14 FLOAT reamostrados24k
  e 14 WAVs PCM16LE mono24k, com hashes individuais. `asr-jobs.json`, `asr.json`,
  `asr-summary.json` e `asr.log` conservam referências e transcrições completas.
- `storage-receipt.json`: contabilidade e limpeza; `baseline-decision.json`:
  decisão compacta e ligações por hash.

O config Cadu tem SHA256
`9433ddafa54e7d5c973d6160a29c046e39d65a2dc8892305591c7b93656c2253`.
O mapa original possui161 símbolos; U+0303 permanece no ID141, com106
ocorrências nos fonemas dos 14 casos. A fonemização usa eSpeak `pt-br`, NFD por
ponto Unicode e `vowel_clusters=None`. O `frontend_check.check_inventory`
existente verificou todos os símbolos antes da inferência; qualquer símbolo
ausente interrompe a execução. BOS1/pad0/EOS2 e o padding entre fonemas seguem
a função `phonemes_to_ids` do runtime.

`PiperVoice.synthesize` aplica `np.clip` mesmo com `normalize_audio=False`.
Para conservar o FLOAT e detectar overshoot, este baseline usa a API oficial
`phoneme_ids_to_audio` (saída bruta de `session.run`), com os mesmos limites de
sentença do frontend. Não aplica ganho, normalização de volume ou limiter.
Concatena as sentenças e reutiliza `resample_pcm` Kaiser beta10 e
`encode_pcm16` existentes. O pico nativo ficou entre0,5198 e0,6371; nenhum
sample ultrapassou a faixa em qualquer estágio. Todos os outputs de sentença
tinham comprimento múltiplo de256; os 14 PCM foram relidos e tiveram os frames
esperados. Isso verifica integridade estrutural; não prova ausência de
omissões semânticas.

O texto `raw` permanece original. `refinement` reutiliza
`training/f5_refine_text.py` (`f5-refinement-1`), sem mudar seus léxicos ou regras.
Os jobs conservam `expected_verbatim`, `expected_spoken` e `expected_effective`
separadamente. Nenhum caso do teste final foi aberto.

## ASR e limites da comparação

Reconhecedor fixo `Systran/faster-whisper-small`, revisão
`536b0662742c02347bc0e980a01041f333bce120`, CPU int8, duas threads, beam5,
idioma pt, temperatura0, sem VAD, prompt ou contexto entre segmentos. Modelo e
arquivos locais foram verificados pelo scorer existente. Comparação remove
somente acentos/caixa/separadores; não reescreve números, nomes ou siglas.

| Frontend, sete casos por variante | WER verbatim | CER verbatim | WER spoken | CER spoken | WER effective | CER effective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 19,259% | 7,152% | 31,818% | 15,632% | 19,259% | 7,152% |
| refinement | 23,704% | 9,801% | 31,169% | 14,439% | 31,847% | 14,744% |

Há diferenças de representação: o ASR retorna `24`, `2,7`, `CPU` e `HTTPS`
onde certas referências usam palavras ou letras separadas. Há também
divergências em palavras comuns. Exemplos do ASR, sem atribuí-los à audição:

- `v_reply`: transcrição exata nas duas variantes.
- `v_pressure` raw: “leio nível”, “14 a galera 5” e “compai”; refinement:
  “leia o nível”, “14 horas e cinco minutos” e “compare”.
- `v_acronyms`: raw inclui “na SOS” e “uma RL”; refinement inclui “CP
  guverificou”, “a PI” e “SUSIN”. Não há ganho geral comprovado do refinement.
- `v_question`: raw coincide com o verbatim após a normalização fixa; refinement
  aparece como “Tessa Feia” no lugar de “terça-feira”.
- `v_sibilants`: ambas aparecem como “teados”/“sessão” em vez de
  “chiados”/“seção”. “Sessão” e “seção” podem ser só escolha ortográfica do ASR.
- `v_long`: há divergências no começo e no nome, mas o reconhecedor preserva
  a cauda “até receber a próxima instrução” nas duas variantes. A cauda
  preservada não prova que todas as palavras anteriores estejam corretas.

São uma inferência estocástica por texto/variante. O seed global ORT foi
fixado, mas não se demonstrou ruído idêntico por request: `v_reply` tem texto e
IDs iguais nas duas variantes, porém dura2,937s e2,879s; `v_sibilants` também
varia com IDs iguais. Portanto, diferenças entre as duas saídas não podem ser
atribuídas somente à reescrita. Não houve audição; ASR não decide sozinho
pronúncia ou timbre.

## Ablação controlada de ruído

O operador autorizou antes da nova avaliação uma hipótese separada: sete
textos `raw`, alterando apenas `noise_scale0,667→0` e `noise_w0,8→0`.
`length_scale1`, modelo, mapa, fonemas, ganho, formato, referências e scorer
continuaram iguais. O config efetivo e seu SHA ficam em
`cadu-zero-noise-cpu/voice-config-effective.json`; as únicas duas diferenças
de configuração foram conferidas. Os IDs fonêmicos dos sete casos são
idênticos aos do baseline raw. O runner original foi preservado com seu SHA
em `base-cadu-cpu/runner-used.py` antes de acrescentar a opção `--zero-noise`.

Os sete WAVs passaram pelos mesmos checks; ASR7/7 terminou às10:10:04 UTC.

| Variante raw, mesmos sete casos | WER verbatim/effective | CER verbatim/effective | WER spoken | CER spoken |
| --- | ---: | ---: | ---: | ---: |
| Ruído original | 26/135 =19,259% | 7,152% | 31,818% | 15,632% |
| Ambos os ruídos0 | 23/135 =17,037% | 6,490% | 30,519% | 15,513% |

A variante zerada passou nominalmente o limiar de WER raw17,56% definido pelo
operador, com melhora de2,222 pontos percentuais sobre esta base. Isso não
constitui ganho geral de clareza: persistem expressões como “14 e a HDR5”,
“na iOS”, “Terça Feia”, “senções” e “vermel” na transcrição. Não foi feita
audição nem atribuída causa acústica por uma síntese.

Um diagnóstico **textual pós-hoc**, com traceback Levenshtein conferido contra
os edits do scorer original, separou referências a nomes, quantidades/datas e
siglas. São5 expressões de nomes,6 numerais/quantidades e6 siglas; quantidades
incluem unidades da frase. Não reescreve o ASR nem modifica o WER usado no gate.

| Word edits do ASR por categoria textual | Ruído original | Ruídos0 |
| --- | ---: | ---: |
| Nomes | 3 | 2 |
| Numerais/quantidades | 6 | 6 |
| Siglas | 2 | 2 |
| Inserções próximas a expressão, associação ambígua | 3 | 0 |
| Fora dessas expressões | 12 | 13 |
| Total | 26 | 23 |

Expressões com edits: nomes3/5→2/5, numerais3/6→3/6 e siglas2/6→2/6. Erros
fora dessas expressões não diminuíram. Mesmo esse grupo pode conter escolha
ortográfica do ASR, como “sessão”/“seção”, e não comprova erro de pronúncia.
Receita reproduzível: `review_alignment.py` e `text-diagnostic.json` na raiz
privada Piper; o relatório registra a associação heurística das inserções.

Como a variante passou o limiar nominal, repetimos os sete textos em processo
e sessão ONNX novos: PID208663→208804. Todos os7 payloads PCM, hashes de WAV
PCM e samples FLOAT nativos ficaram bit a bit iguais. Prova completa em
`zero-noise-reload-proof.json`. Isso demonstra replay determinístico nessas
condições; não prova que o ruído causou as diferenças de ASR ou que a
pronúncia/identidade melhorou. Nenhum ASR adicional foi necessário porque o
áudio de entrada era idêntico.

As21 cópias de áudio redundantes do reload foram removidas somente depois de
verificar proprietário, ausência de FDs ativos e equivalência; recuperados
12,77MB. As primeiras cópias selecionadas, configs, receitas, hashes e recibo
do reload permanecem. `cadu-zero-noise-cpu-reload/cleanup-receipt.json` aponta
cada cópia mantida. O wheel removido anteriormente recuperou outros34,13MB.
Antes da etapa posterior de checkpoint completo de outro agente, Piper
ocupava151,70MB e o ledger global36,597GB. A ablação e o ASR terminaram antes
da janela Vox prevista para10:20; não usaram GPU ou decodificação concorrente.

Repetição da hipótese, em run novo:

```sh
CUDA_VISIBLE_DEVICES='' \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_piper_base_cpu --frontends raw --zero-noise \
  --run /var/tmp/aegis-v2-work/piper/cadu-zero-noise-cpu-repeat
```

## Controle de frontend com os dois ruídos zerados

O refinement também foi executado com0/0, em sete casos. O controlador local
PID209429 esperou a remoção de `pause-asr`, ausência de `active.json`,
`completed_probes≥1` e ausência de outro decoder/ASR. O primeiro probe
FastPitch4000 terminou antes da liberação; os sete ASRs do refinement foram
concluídos às10:37:11 UTC. Cinco testes stdlib temporários verificaram esses
gates sem carregar modelo. Nenhum helper/runner compartilhado foi alterado.

Artefatos em `cadu-zero-noise-refinement-cpu/`; supervisão e gates em
`refinement-zero-control/`. O runner herdava um rótulo textual FP077 antigo;
`asr-jobs-controlled.json` conserva o contexto verdadeiro de execução após
qualidade4000. Isso não cria uma medição de latência de produção.

| Controle0/0, sete casos | WER verbatim | CER verbatim | WER spoken | CER spoken | WER effective | CER effective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 17,037% | 6,490% | 30,519% | 15,513% | 17,037% | 6,490% |
| refinement | 19,259% | 8,212% | 27,922% | 14,439% | 29,299% | 14,744% |

O refinement piora o WER verbatim agregado, mas a intenção de14h05 fica mais
bem representada na transcrição: “Aegis, leia o nível de pressão às14 horas e5
minutos e compare com2,7 quilopascais.” O raw0/0 apareceu como “Agis, leio
nível de pressão às14 e a HDR5 compara com2,7 quilopascais.” Essa observação
não é avaliação auditiva. As referências spoken/effective continuam separadas;
o scorer não converte os dígitos do ASR em palavras. Nenhum frontend foi
promovido por esse único agregado ou caso.

| Caso | Transformação efetiva | IDs fonêmicos iguais? | PCM igual? | Word edits verbatim raw→refinement |
| --- | --- | --- | --- | ---: |
| `v_command` | Aegis→A éjis | não | não | 1→0 |
| `v_reply` | nenhuma | sim | sim | 0→0 |
| `v_pressure` | nome,14h05→quatorze horas e cinco minutos,2,7→dois vírgula sete | não | não | 7→5 |
| `v_acronyms` | léxico explícito de letras/siglas | não | não | 5→9 |
| `v_question` |24→vinte e quatro | não | não | 1→0 |
| `v_sibilants` | nenhuma | sim | sim | 2→2 |
| `v_long` | Aegis→A éjis | não | não | 7→10 |

Em `v_question`, a suposição de IDs iguais não se confirmou: o eSpeak altera
`ɪʲ→i` e acrescenta um espaço ao converter24 literal versus “vinte e quatro”.
O relatório conserva o contexto exato em símbolos. Assim, PCM diferente nesse
caso é esperado. Todos os casos que realmente têm IDs iguais (`v_reply` e
`v_sibilants`) produziram PCM bit a bit igual. O controle remove a confusão de
ruído desses pares, sem provar pronúncia ou identidade.

`raw-refinement-controlled-comparison.json` e
`frontend-zero-case-review.json` conservam tokens de texto, diffs fonêmicos,
hashes PCM e os três textos esperados por caso, além de ambas as transcrições
ASR e seus scores. Siglas continuam problemáticas no ASR; a cauda longa
refinada apareceu como “A ESP menacerá”. O raw0/0 fica como referência
provisória primária; refinement0/0 permanece secundário, sem vencedor geral.
Qualquer uso posterior do refinement deve compará-lo com o mesmo frontend,
config0/0, corpus e referências **antes e depois** da especialização.

Repetição somente após a janela CPU ser liberada e o probe FP prioritário
terminar, usando run novo:

```sh
CUDA_VISIBLE_DEVICES='' \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_piper_base_cpu --frontends refinement --zero-noise \
  --run /var/tmp/aegis-v2-work/piper/cadu-zero-noise-refinement-cpu-repeat
```

## Decisão e retomada

Passaram carga ONNX real, mapa161 com nasalização, saída FLOAT íntegra e PCM
estrito em CPU. A base não está aprovada para substituir a Aegis: não tem sua
identidade treinada, não houve benchmark local isolado de produto, e há
divergências ordinárias de PT-BR a resolver. Nenhum frontend venceu de forma
geral nesta execução. A licença/linhagem MIT/GPL/Lessac permanece conforme a
pesquisa anterior: apenas pesquisa local privada, sem inferir permissão de
distribuição ou uso final.

Próxima ação concreta, antes de treino longo: verificar as divergências de
frontend e a receita pinada; o ruído zerado fica como controle provisório
reproduzível, sem aprovação de pronúncia/identidade. O operador autorizou outro
agente a adquirir somente o checkpoint Cadu completo e preparar carga estrita
G/D e reconstrução de âncoras reservadas, seguida de3updates→save→novo processo
→update4. A revisão independente da retomada/Adam/RNG do trainer histórico
continua necessária. Nenhum checkpoint845MB foi baixado por este subtask;
a aquisição posterior pelo outro agente tem recibo e contabilidade próprios.
A preparação atual do novo loader é somente CPU; reconstrução e piloto GPU
aguardam a janela serial controlada pelo operador.

Repetição em run novo (somente CPU; exige o overlay/modelo verificados):

```sh
CUDA_VISIBLE_DEVICES='' \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_piper_base_cpu \
  --run /var/tmp/aegis-v2-work/piper/base-cadu-cpu-repeat
```

O script impõe nice19 e threads ONNX2/1. Depois de terminar a síntese e conferir
ausência de outro decoder/ASR:

```sh
CUDA_VISIBLE_DEVICES='' nice -n19 \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  benchmarks/v2/asr_score.py \
  --jobs /var/tmp/aegis-v2-work/piper/base-cadu-cpu-repeat/asr-jobs.json \
  --output /var/tmp/aegis-v2-work/piper/base-cadu-cpu-repeat/asr.json
```

Extração medida143,27MB de insumos simultâneos; pico conservador200MB incluindo
áudios. Após remover o wheel redundante, a árvore Piper ocupa138,15MB alocados
e foram recuperados34,13MB. Ledger global deduplicado após limpeza36,583GB,
disco livre27,668GB. O ledger inclui todos `/var/tmp/aegis*`, ZIP de origem,
worktree experimental e cache do usuário, sem os pesos/imagem de produção
preexistentes. A fonte de dados original e arquivos de outros processos foram
preservados.
