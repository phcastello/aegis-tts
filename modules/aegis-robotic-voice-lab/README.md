# Aegis robotic voice lab

Laboratório isolado: Chatterbox pt-BR é somente o articulador GPU. WORLD e LPC
ressintetizam integralmente no CPU; nenhum candidato estrito recebe ou mistura o
waveform humano de origem. Não é integração de produção.

## Repetição

```bash
python3 modules/aegis-robotic-voice-lab/scripts/aegis_robotic_voice.py prepare
bash modules/aegis-robotic-voice-lab/scripts/create_environments.sh
python3 modules/aegis-robotic-voice-lab/scripts/aegis_robotic_voice.py preflight
python3 modules/aegis-robotic-voice-lab/scripts/aegis_robotic_voice.py run-mvp
```

O ambiente Chatterbox fica em `private/environments/chatterbox`; o DSP fica em
`private/environments/dsp`. Ambos são privados e o script nunca toca no Python
global ou no ambiente vLLM.

## Qwen 0.6B frio + efeitos selecionados

O executor abaixo é propositalmente **manual**: ele não foi executado durante a
implementação. Usa `Aninha_AegisVoice4.wav` e a transcrição correspondente,
sobe somente um contêiner vLLM de laboratório em `127.0.0.1:18001`, gera o
corpus completo para Q00–Q04 e, em seguida, aplica somente os dois caminhos
escolhidos: `L00` (source-filter sintético) e `W04` (WORLD extremo). Ele não
toca nem para o serviço de produção existente.

```bash
bash modules/aegis-robotic-voice-lab/scripts/run_qwen_cold_lab.sh
```

Os cinco perfis são explícitos em `config/qwen-cold.yaml`, não um produto
cartesiano: Q00 é quase determinístico (`temperature 0.05`, `top_p 0.30`,
`top_k 1`, penalidade `1.15`) e Q01–Q04 aumentam a variação apenas de forma
limitada. Em vLLM-Omni 0.22 esses controles pertencem ao deploy Stage 0; por
isso o script encerra apenas seu próprio contêiner entre perfis, preserva o
cache e registra o arquivo de deploy usado no manifesto.

Para uma inspeção curta antes do lote total:

```bash
bash modules/aegis-robotic-voice-lab/scripts/run_qwen_cold_lab.sh --profiles Q00
```

Saídas e relatórios privados ficam em `results/qwen-cold/` e
`reports/generated/qwen-cold/`. Os candidatos `L00` e `W04` são ressintetizados
integralmente; o WAV fonte Qwen nunca entra como camada audível ou dry mix.

O caminho selecionado para polimento é `Q00 → W04`; seus parâmetros completos,
hashes de referência e métricas verificadas estão em
`config/qwen-selected-q00-w04.yaml`.

## Polimento Q00 → W04 v0.2

O polimento de sibilância reutiliza exclusivamente os 15 WAVs Q00 congelados.
Ele não inicia Qwen, vLLM ou Docker e recusa qualquer caminho sob
`protected/AegisVoicev0.1`. A identidade W04 (F0, quantização de 200 cents,
transição zero, formantes 1.10 e ressíntese WORLD estrita) fica bloqueada no
catálogo; somente o envelope de regiões não vozeadas/de transição e o de-esser
do áudio ressintetizado podem atuar.

```bash
bash modules/aegis-robotic-voice-lab/scripts/run_qwen_w04_polish.sh \
  --config modules/aegis-robotic-voice-lab/config/qwen-w04-sibilance-polish-v0.2.yaml
```

O comando renderiza sequencialmente `S01`, `S02` e `S03`, usando hashes para
confirmar as fontes e cache estrito para nunca sobrescrever resultado divergente.
Os WAVs ficam em `results/qwen-polish-v0.2/{raw,mastered}/Sxx/`; a página de
escuta em `results/qwen-polish-v0.2/reports/listening.html` traz a referência
externa Q00 → W04, raw e mastered de cada variante. `metrics.json` registra a
energia de alta frequência das frases diagnósticas, sem escolher vencedor.

## Comparação focal v0.3

Para uma escuta com diferenças mais evidentes, a matriz focal usa somente
`sibilants`, `numbers`, `long` e `sarcastic`, com dez correções graduais de
sibilância. Ela não aplica correção de nasalidade: as frases com muitas nasais
servem para revelar mudanças indesejadas de identidade.

```bash
bash modules/aegis-robotic-voice-lab/scripts/run_qwen_w04_sibilance_focus.sh \
  --config modules/aegis-robotic-voice-lab/config/qwen-w04-sibilance-focus-v0.3.yaml
```

O comando gera 40 candidatos em
`results/qwen-polish-v0.3-sibilance-focus/`, atualiza o manifesto e as
métricas, e cria `reports/listening.html` (referência + 10 versões mastered)
e `reports/technical-stages.html` (raw vs. mastered). Ele não inicia Qwen,
vLLM ou Docker.

## Comparação de aperiodicidade vozeada v0.4

Esta rodada mantém a base W04 em `0.20`, mas testa `0.14`, `0.17`, `0.20`,
`0.23`, `0.26` e extremos conservadores entre `0.12` e `0.30`. Só a
aperiodicidade dos frames vozeados muda; F0, quantização, formantes e o
tratamento não vozeado continuam bloqueados.

```bash
bash modules/aegis-robotic-voice-lab/scripts/run_qwen_w04_aperiodicity_focus.sh \
  --config modules/aegis-robotic-voice-lab/config/qwen-w04-aperiodicity-focus-v0.4.yaml
```

Como as rodadas anteriores, o script cria ou atualiza os WAVs, manifestos,
métricas e páginas HTML sob `results/qwen-polish-v0.4-aperiodicity-focus/`.

## Teste agressivo de sibilância v0.5

Quando as variações moderadas não são perceptíveis, esta rodada reduz o
envelope de regiões não vozeadas em até 36 dB acima de 5,2 kHz, com transição
curta, e aplica de-essing de até 28 dB. É um teste de limite deliberadamente
agressivo para tornar a diferença inequívoca, não uma recomendação de mix.

```bash
bash modules/aegis-robotic-voice-lab/scripts/run_qwen_w04_sibilance_aggressive.sh \
  --config modules/aegis-robotic-voice-lab/config/qwen-w04-sibilance-aggressive-v0.5.yaml
```

Os resultados e páginas de escuta são criados em
`results/qwen-polish-v0.5-sibilance-aggressive/`.

## Proveniência fixada

O loader local deriva do Space oficial pt-BR `9e515821e826e207cd617a0fdd0223899ed108ea`
e da revisão do código `5de7a54aa4e5e2baadb0182dde554908b48b85c2`. Ele fixa o
modelo `ResembleAI/Chatterbox-Multilingual-pt-br` na revisão
`b3952f18bc2eaa72b9bd7c17d2c4653bcad4770d`, locale `pt-BR`, id `pt`, T3
`t3_pt_br.safetensors`, decoder V3 e tokenizador multilíngue. Veja hashes em
`config/chatterbox.yaml`. A licença declarada pelo modelo é MIT.

O Space oficial usava o `s3gen_v3.pt`; este laboratório usa seu equivalente
oficial `s3gen_v3.safetensors` e o carrega diretamente, evitando pickle.

Os relatórios são evidência experimental: não declaram êxito sem áudio gerado,
medição real, e escuta humana. `promote` limita o long-form a até três candidatos.
