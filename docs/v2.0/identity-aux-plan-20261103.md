# Evidência auxiliar de identidade — preparo CPU, 2026-11-03

Hipótese: descritores acústicos e um encoder já local podem localizar diferenças
entre sistemas e orientar poucos pares de escuta; não podem aprovar a identidade
robótica, compensar pronúncia ruim ou escolher a vencedora.

`training/f5_acoustics.py` já oferece YIN com máscara de energia, curvas de F0 e
espectro; reutiliza `texture_metrics` de `training/reconstruct.py`. O inventário
não encontrou pesos de verificação de locutor independentes nos ambientes F5 ou
dados locais auditados. O F5 inclui código ECAPA-TDNN, sem seus pesos: não será
tratado como encoder treinado. O histórico F5 usa também referência de treino;
ela não substitui as três âncoras reservadas deste experimento.

`training/v2_identity_aux.py` usa somente as três âncoras de validação
`aegis_ptbr_0645`, `aegis_ptbr_0040`, `aegis_ptbr_0910`, o PCM final já arquivado
das sete frases de validação e repetição zero. O teste final permanece fechado.
Todos os WAVs ficam vinculados aos hashes dos recibos. Não copia áudio nem
transcrições privadas para o repositório. A inclusão de uma base Piper de idioma
é marcada como base, sem apresentá-la como finalista especializada.

O Qwen local contém 76 tensores `speaker_encoder.*`, 8.854.336 parâmetros BF16
(17.708.672 bytes), num encoder ECAPA-like 1024D de 24 kHz. É possível carregar
apenas esse prefixo em FP32 na CPU, sem TTS, tokenizer, dependência Transformers,
GPU ou download. O helper compila as seis classes e as duas funções de mel
inalteradas da fonte pinada; o `load_state_dict` é estrito. Mel: FFT1024,
janela1024, hop256, Slaney128, 0–12 kHz, reflect e log, iguais ao extrator
upstream. Fontes e pesos são Apache-2.0:
[código pinado Qwen](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py),
[pesos Qwen locais correspondentes](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/5d83992436eae1d760afd27aff78a71d676296fc).
O encoder condiciona o próprio Qwen: há viés em favor dessa referência e sua
similaridade não equivale a um verificador independente nem mede caráter metálico.

O inventário registra o SHA integral do modelo conferido anteriormente, identidade
de arquivo/tamanho/inode e um novo hash somente dos tensores selecionados. Não
rele os 1,83 GB na janela de medição. Caminho local do modelo:
`/home/phc/aegis-tts/models/Qwen3-TTS-12Hz-0.6B-Base/model.safetensors`.
O código isolado está em `/var/tmp/aegis-v2-work/qwen-sft-research/qwen_tts/core/models`.

Na análise, todos os sistemas recebem o mesmo alvo de RMS ativo, com ganho linear
e um alvo comum reduzido quando necessário para manter todos os picos ≤0,95.
Isso não mede LUFS nem garante volume percebido igual; os originais não mudam.
Reportam-se níveis nativos, clipping, duração, F0/YIN com incerteza de vozeamento,
centroide, flatness, energia 4–8/8–12 kHz, cosseno contra cada âncora e centroide
das três. Não há limiar nem classificação automática. As âncoras têm textos
diferentes das frases geradas: cada comparação registra essa confusão. A
variabilidade dos três pares entre âncoras é contexto, sem intervalo estatístico
representativo da voz inteira.

DTW opcional fica restrito a entradas com o mesmo hash do texto solicitado.
Usa log-mel32, hop20ms, caminho completo e banda25%, no máximo2000 frames e
500 mil células por par. Registra separadamente se o texto efetivo é igual ou
desconhecido: diferenças de frontend, idioma e fonética podem dominar a distância.
Não alinha textos diferentes das âncoras nem atribui essas diferenças à identidade.

O passe é limitado a40 clipes mono de0,2–40s, dois threads, nice19, sem novos
pesos ou áudios derivados. Orçamento: ~35 MB para encoder FP32 e expectativa
de poucos MB de recibos/vetores; a RAM total inclui bibliotecas Torch/librosa.
O helper recusa iniciar/continuar análise se `pause-asr` existir e nunca inicializa
CUDA. O controle pausa permite ao root reservar CPU ociosa para os benchmarks.

Comandos, no worktree experimental, após liberar a janela de CPU:

```bash
PY=/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python
$PY -m training.v2_identity_aux inventory
$PY -m training.v2_identity_aux manifest \
  --benchmark-dir /var/tmp/aegis-v2-work/bench/qwen-warm \
  --benchmark-dir /var/tmp/aegis-v2-work/bench/f5-nfe16
$PY -m training.v2_identity_aux analyze \
  --manifest /var/tmp/aegis-v2-work/identity-aux/manifest.private.json \
  --output /var/tmp/aegis-v2-work/identity-aux/common-validation \
  --dtw-cases v_command v_sibilants
$PY -m unittest tests.test_v2_identity_aux -v
```

Uma finalista com `manifest.json/results.jsonl` comuns pode ser acrescentada com
`--benchmark-dir`, mantendo a saída em nova pasta. `--piper-dir` aceita recibos
CPU da base Cadu, identificados separadamente. A análise não serve para medir
latência: seus tempos próprios são apenas custo do laboratório. O A/B humano
continua separado, com clipes do mesmo texto e chave privada; nenhum resultado
de escuta é inventado pelo helper.

O NFE8 especializado contém uma falha real de conversão PCM em `v_question`
na repetição zero. O gerador de manifest exige sete produtos completos e recusa
essa matriz; não apresenta seu FLOAT de depuração como áudio entregue ao usuário.
O NFE16 especializado fornece a referência F5 completa deste primeiro passe.

Prova CPU após liberação da pausa às 11:49: oito testes passaram (0,027 s de
execução dos fixtures). O encoder real carregou estritamente os 76 tensores e
produziu vetores finitos de 1024 dimensões para as três âncoras, com análise F0/espectral:
7,281 s, RSS máximo 779,49 MiB, dois threads, nice19 e CUDA não inicializada.
Não houve escuta nem comparação numérica das candidatas neste smoke test.
Recibos: `/var/tmp/aegis-v2-work/identity-aux/cpu-fixture-tests.json` e
`/var/tmp/aegis-v2-work/identity-aux/cpu-proof-anchors/results.json`.
Todos os artefatos auxiliares mantidos somam cerca de 79 kB, sem cópia de pesos.

O manifest privado derivado `identity-aux/manifest-roundtrip-proposed.private.json`
contém 23 itens: o passe comum de 17 mais seis reconstruções acústicas das mesmas
âncoras. Confere os WAVs e a cadeia de origem: Vox AudioVAE48k→16k→48k e Piper
original48k→prepared24k→22,05k→posterior/decoder. Cada reconstrução preserva
`source_reference_sha256` e `text_group`, além dos recibos e comprimentos. São
FLOATs nativos, sem apresentá-los como síntese condicionada por texto ou produto.
O passe foi executado em CPU após liberar a janela: 23 clipes em 16,272 s,
sem inicializar CUDA. Recibo privado: `identity-aux/baselines-roundtrips/results.json`.
A mediana do cosseno ao centroide das âncoras foi 0,9819 no Qwen processado,
0,9900 no F5 especializado NFE16, 0,9937 nas três reconstruções AudioVAE e
0,9781 nas três reconstruções posteriores Piper. As reconstruções usam o
mesmo texto das âncoras, enquanto as sínteses usam textos diferentes; o encoder
é do Qwen e não tem calibração. Esses números não ordenam identidade vocal
nem aprovam qualquer sistema. Não são um teto: uma adaptação posterior pode
mudar tanto o codec quanto a distribuição de entrada.
