# Controle proposto: Vox com a referência TRAIN0586 do F5

Hipótese: a referência acústica fixa0586 pode reduzir a variação do timbre do
LoRA800/1400 em texto novo, preservando conteúdo e a vantagem de espera. Há
rota oficial para este controle; seu efeito ainda não foi medido. Esta revisão
carregou somente tokenizer/áudio/mel em CPU2, sem modelo TTS, CUDA ou fila.

## Referência e vínculo com o F5

`aegis_ptbr_0586` pertence ao treino, grupo `tema20_grupo04`; sua ausência e a
do grupo na validação foram conferidas. Não foram lidos target50/teste final6.
O áudio preparado tem10,103s, mono24kHz/PCM24,242.472samples. É exatamente o
arquivo que originou a referência948frames do benchmark F5 especializado
NFE16; áudio, mel e texto conferem com seu manifest. O texto fica no recibo
privado, preservando caixa original e versão F5 em minúsculas.

| Insumo | SHA-256 |
| --- | --- |
| WAV preparado24k, também usado pelo F5 | `197b86b91143efa8ca83b07af7eb0ea26b16b9253dcf0c8a17fb0531d94c5397` |
| WAV original48k | `96990f7cf16074bf0541db70a18690919912afce6dffc9ff7f8709a1c08f7da0` |
| Mel F5100×948 | `00c5efa97c56c2355adc852588d9293d4b2a423c0d4e443f648c98f4b7b3b6d3` |
| Texto efetivo da referência F5 | `d8d3ffcb1cd09acfb43218eea582a4f73dae155f2ebcc3d0990ab0c721c1d17f` |

O WAV é finito, pico0,53256, RMS−22,913dBFS e zero samples de módulo≥1.
Quadros20ms com RMS>−45dBFS começam em0,14s e terminam em9,92s: são medidas
de energia, sem classificação de fala ou escuta. Manter a fala inteira,
sem ganho, VAD, corte, WORLD ou denoiser. O ASR histórico dessa fonte marcou
4/25 edições, incluindo divergências de palavras comuns. Isso não prova erro
da transcrição ou do áudio e desaconselha assumir alinhamento perfeito para
continuation. O controle inicial usa apenas a referência acústica.

## API oficial e caminho de execução

Fonte local limpa no commit `cce58a5b59303c9bd63d12f23afe6a49a4a80c59`,
pesos locais na revisão `32279effe8c19989596f05d353d1447f51d9e915`.
As [APIs oficiais](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/model/voxcpm2.py#L694)
permitem três modos:

- `build_prompt_cache(reference_wav_path=wav, trim_silence_vad=False)`:
  modo `reference`, prefixo isolado `[103, características da referência,104]`.
  A geração recebe somente o texto novo; não recebe a legenda0586.
- `prompt_text` + `prompt_wav_path`: `continuation`, concatenação direta
  de legenda e texto novo, seguida pelas características do áudio. Não existe
  espaço implícito; a tokenização deve contar a concatenação exata.
- Os três argumentos: `ref_continuation`, com ambos os prefixos. É uma
  ablação posterior possível, após esclarecer o alinhamento da legenda.

Construir uma vez por processo/checkpoint e guardar o cache fixo fora do
objeto do modelo. Chamar `generate_with_prompt_cache_streaming(...)` com o
mesmo cache em cada pedido; cada item é uma tupla `(wav,tokens,features)` e
somente `item[0]` é áudio. Não usar `merge_prompt_cache`: acumula texto/áudio
entre pedidos e altera tanto contexto quanto orçamento.

`build_prompt_cache` já usa `inference_mode`. `_encode_wav` lê pelo librosa,
reamostra24→16k na CPU, preenche à direita e executa o encoder VAE na GPU.
Devolve características na CPU. A prévia local com librosa0.11.0 deu161.648
samples16k; o padding de2.192samples leva a163.840 e prevê64patches
`[64,4,64]`,65.536bytes FP32. A forma real precisa ser confirmada no piloto.
Não transferir esse dict inteiro à GPU: o empacotador oficial cria tokens e
zeros em CPU antes de concatenar. O caminho transfere o lote de entrada
para a GPU por pedido; pesos, VAE FP32 e KVs continuam residentes na GPU.
Esse staging de entrada não descarrega pesos na CPU.

O cache evita reler/reamostrar/codificar a referência em cada pedido, mas
**não** reutiliza KV ou prefill. As duas LMs processam todo o prefixo em cada
chamada. O prefixo de referência aumenta66posições; atenção/prefill e H2D
entram no TTFA. O VAE streaming começa sem contexto anterior no modo
`reference`; sua saída contém apenas os novos patches. Continuation usa até
três patches anteriores e tem tratamento de contexto diferente entre arquivo
e streaming; não misturar esses modos no primeiro controle.

## Limite exato de contexto

Defina T=tokenização nativa do texto novo; C=tokenização nativa da legenda
com separador explícito mais texto novo; R/P=patches de referência/continuação;
M=`max_len` pedido. Prefixos L são T+1, R+T+3, C+P+1 e R+C+P+3,
respectivamente. A reserva conservadora é **L+M+1≤1024**, incluindo uma
posição de margem: mesmo a última iteração limitada por M pode executar
`forward_step`. Reservar M400 inteiro, embora o limite nativo efetivo seja
`min(6*T+10,400)`. Recusar excesso; não reduzir M nem cortar texto.

Com R64, o modo escolhido precisa T+468slots. As contagens usam exatamente
`LlamaTokenizerFast` mais o wrapper chinês da fonte; reproduzem os sete casos
do benchmark nativo anterior. A advertência do nome `VoxCPM2Tokenizer` no
config também ocorre no loader oficial; não foi usado outro tokenizador.

| Validação | T original / números | Prefixo ref original / números | Reserva ref original / números |
| --- | ---: | ---: | ---: |
| v_command | 15 /15 | 82 /82 | 483 /483 |
| v_reply | 14 /14 | 81 /81 | 482 /482 |
| v_pressure | 35 /44 | 102 /111 | 503 /512 |
| v_acronyms | 25 /25 | 92 /92 | 493 /493 |
| v_question | 31 /35 | 98 /102 | 499 /503 |
| v_sibilants | 27 /27 | 94 /94 | 495 /495 |
| v_long | 145 /145 | 212 /212 | 613 /613 |

Até combined com a legenda original+espaço prevê728slots na longa. O guard
atual `v2_voxcpm_cache_probe.capacity_hooks` cobre apenas `_generate` vazio
e **não intercepta `_generate_with_prompt_cache`**. Um piloto futuro precisa
guard próprio dessa entrada, assert dos dois KVs1024 e da forma do cache.
Nenhuma chamada com referência deve passar como coberta pelo guard atual.

## Piloto pequeno, ainda não implementado/enfileirado

Começar no LoRA800 preservado; repetir a mesma matriz no1400 se houver sinal
útil sem regressão clara de conteúdo/velocidade. Usar `v_command`,
`v_pressure` e `v_long`, o frontend fixo `ptbr-numbers-v1`, NFE10, CFG2,
seed40733133, max_len400, retryFalse, compilação oficial, TF32/precisão iguais
ao controle1024 e o mesmo FIR48→24k/PCM16. Não fundir LoRA nem alterar pesos.

Para cada checkpoint, um processo novo carrega a cópia salva estritamente,
confere SHA/residência completa e alterna A=`prompt_cache=None` e
B=`reference0586`. Usar a entrada cacheada em **ambos** evita confundir a
troca de API com a referência. Antes de comparar B, conferir A contra a rota
zero-shot anterior, duração/hash/limite e cancelamento/fechamento do gerador.
Medir o cache build frio separadamente, com sincronização CUDA e pico VRAM;
se for lazy no produto, esse custo entra na primeira fala fria.

Registrar primeira compilação separada; três warmups excluídos por caso/modo,
seguidos de n3 medidos por caso/modo, com ordem A/B alternada. São18pedidos
medidos por checkpoint,36 para ambos, além dos warmups. Medir no cliente
nativo primeiro PCM, primeira fala pela regra comum, EOF, RTF, gaps, cap/EOS,
hashes brutos/PCM e replay real dos graphs. Guardar FLOAT/PCM de rep0 e falhas;
n3 com a mesma seed mede variação temporal, não diversidade de conteúdo.

Avançar se a longa continuar RTF<1, não houver falha/truncamento/overshoot,
o PT dos pares não regredir na análise ASR+casos divergentes e os descritores
de identidade motivarem escuta. Cosseno/F0/espectro não são aprovação nem
limiar. Não condicionar nas três âncoras reservadas: servem somente à análise.
O piloto nativo imediato não replica o gateway: se a referência for útil,
medir depois o serviço real com startup320ms e exigir gaps0 e vantagem TTFA
material sobre o Qwen. Não somar320ms artificialmente ao relatório nativo.

Sem novos pesos/downloads/cópia de ambiente. Os CPs800/1400 já estão nos
snapshots de identidade; manter hardlinks e hashes. Reservar128MiB adicionais
para WAVs/traços/cache/receitas; pico de VRAM do encoder e do prefixo ainda
exige medição, embora a capacidade dos KVs permaneça1024. Antes da execução,
recalcular o ledger40GB e margem5GiB. Último ledger do passe1400:37,765GB;
o próprio estudo novo ocupa menos de1MiB.

Recibo privado e receita CPU testada:
`/var/tmp/aegis-v2-work/voxcpm2/reference0586-control/{cpu-audit.json,cpu-research.py,pilot-proposal.private.json}`.
Ele contém os caminhos e textos privados, hashes da fonte/dados, contagens
completas dos quatro modos e prova `cuda_initialized=false`.
