# F5-TTS PT-BR: evidência reproduzível na Aegis

Estado em 02/11/2026: **checkpoint e vocabulário verificados, Vocos reconstruído
em CPU com áudio tratado real, três sínteses locais concluídas em NFE 8/16/32**.
Foi sintetizado um texto da coorte fixa, com uma seed e uma observação por NFE.
Nenhuma carga CUDA,
treino, mudança de endpoint ou interrupção da produção foi feita neste ensaio.
A escolha como substituta ainda depende de pronúncia, escuta, preservação dos
efeitos e medições na A2000. A saída de 24 kHz não elimina a candidata.

## Artefatos e licença

O [card brasileiro fixado](https://huggingface.co/firstpixel/F5-TTS-pt-br/blob/0af67bc061699cab46218c150948910fde209614/README.md)
declara treinamento brasileiro com centenas de horas e clonagem por referência.
Isso é relato do autor, sem validação independente da composição ou dos direitos
de cada fonte de treinamento. A licença dos pesos é **CC BY-NC 4.0**, com
restrição de uso comercial; o modelo não deve ser descrito como comercialmente
irrestrito. O [código F5](https://github.com/SWivid/F5-TTS/blob/45868abaabd74aabf03cd673d6dd00cc9d9c01ae/LICENSE)
é MIT. Código e pesos [Vocos](https://huggingface.co/charactr/vocos-mel-24khz/tree/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21)
são publicados como MIT. As licenças permissivas das ferramentas não removem a
restrição NC do checkpoint PT-BR.

| Componente | Pin / tamanho | SHA-256 validado localmente |
| --- | --- | --- |
| `firstpixel/F5-TTS-pt-br`, `pt-br/model_last.safetensors` | revisão `0af67bc061699cab46218c150948910fde209614`; **1.348.431.976 bytes** | `c25e2678bdd4304071e3d0e39a283a7d23ffbc23678080c5b0b741dda2e07f0f` |
| Código `SWivid/F5-TTS` | commit limpo `45868abaabd74aabf03cd673d6dd00cc9d9c01ae`; versão declarada 1.1.22 | fonte conferida pelo commit Git |
| Vocabulário HF `SWivid/F5-TTS`, `F5TTS_Base/vocab.txt` | 13.800 bytes, CRLF, 2.545 entradas | `2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c` |
| Vocabulário `src/f5_tts/infer/examples/vocab.txt` da fonte fixada | 11.255 bytes, LF, mesmas entradas na mesma ordem | `4e173934be56219eb38759fa8d4c48132d5a34454f0c44abce409bcf6a07ec46` |
| `charactr/vocos-mel-24khz`, `pytorch_model.bin` | revisão `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`; 54.365.991 bytes | `97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a` |
| Vocos `config.yaml` | mesma revisão; 461 bytes | `da9033922f969a47f0c160010226919e59f27761fd5066f3828d46de6650b0fc` |

O manifesto Hub foi consultado antes do download e o hash anunciado foi comparado
com o conteúdo completo. Não foram baixados `model_200000.pt` nem
`model_last.pt`, de aproximadamente 5,39 GB **cada**, nem o repositório inteiro.
O checkpoint acústico é safetensors. O Vocos oferece aqui um state dict `.bin`
oficial, fixado e validado; a leitura utiliza `torch.load(..., weights_only=True)`
e `load_state_dict(..., strict=True)`.

Os recibos e arquivos ficam exclusivamente em
`/var/tmp/aegis-v2-evaluation-20261102/f5/`: `hub-manifest.json`,
`checkpoint-receipt.json`, `tensor-header.json`, `swivid-manifest.json`,
`commit-history.json`, `probe-receipt.json`, `source/`, `vocos/` e `reference/`.
Transcrições, pesos e WAVs não são entregáveis Git. As permissões privadas são
0700 para diretórios e 0600 para os artefatos gerados.

O [manifesto mínimo](../../benchmarks/architecture_compare/artifacts.f5.json)
registra exatamente os quatro downloads necessários, inclusive o pin do
vocabulário HF `84e5a410d9cead4de2f847e7c9369a6440bdfaca`. Para reproduzir,
consultar `/api/models/{repo}/revision/{revision}?blobs=true`, comparar SHA e
tamanho anunciados com o manifesto e baixar somente
`https://huggingface.co/{repo}/resolve/{revision}/{path}` para a raiz privada.
Validar o SHA-256 do arquivo completo antes do probe; não carregar `.pt`
históricos nem substituir o `resolve` por `main`.

O [freeze do ambiente](../../benchmarks/architecture_compare/requirements-f5-cu118.lock.txt)
e o [inventário de licenças declaradas](../../benchmarks/architecture_compare/environment-f5-licenses.json)
separam versões de pacotes e licença dos pesos. `encodec==0.1.1`, dependência
instalada do pacote Vocos, declara CC BY-NC 4.0 em seus metadados; nenhum peso
Encodec foi baixado ou usado no caminho mel Vocos. As wheels CUDA têm seus
próprios termos NVIDIA. O inventário não é certificação de direitos comerciais.

## Arquitetura e compatibilidade

O cabeçalho safetensors contém 364 tensores FP32, somando 337.096.836 escalares,
incluindo o buffer rotativo de 32 escalares. O embedding de texto tem shape
`[2546,512]`: 2.545 tokens mais a posição reservada pelo código. Há 22 blocos,
largura de 1.024, projeção mel de 100 canais e convoluções de texto de 512 canais.
Essas dimensões são coerentes com **F5TTS_Base**. Os tensores foram publicados
sem prefixo `ema_model.`; o runner aceita esse formato e exige carga estrita.

Não se deve usar o default atual `F5TTS_v1_Base` da API. O histórico brasileiro
termina em **04/02/2025**, e o upload dos pesos é de **28/01/2025**, anterior ao
[anúncio oficial do v1 em 12/03/2025](https://github.com/SWivid/F5-TTS/blob/45868abaabd74aabf03cd673d6dd00cc9d9c01ae/README.md).
As configurações fixadas também diferem em opções que **não mudam todos os
shapes**: Base usa `text_mask_padding=False` e `pe_attn_head=1`; v1 usa
`text_mask_padding=True` e `pe_attn_head=null`. Portanto, compatibilidade de
dimensões sozinha não prova equivalência semântica. O probe instancia
explicitamente a configuração Base antiga, com 16 heads, `ff_mult=2`, quatro
camadas convolucionais e atenção PyTorch. A carga **estrita passou**, com
**337.096.804 parâmetros FP32** e sem ignorar buffers, renomear tensores ou
aceitar chaves faltantes. O recibo final registra esse resultado.

A identificação também foi conferida contra a [API oficial histórica](https://github.com/SWivid/F5-TTS/blob/33e865120c1ea08c39e9fb7b68e2ebd09b37ce58/src/f5_tts/api.py),
commit `33e865120c1ea08c39e9fb7b68e2ebd09b37ce58`, anterior ao fim das alterações
do card brasileiro. Seu default `F5-TTS` instancia exatamente largura 1.024,
22 blocos, 16 heads, FF2, texto 512 e quatro convoluções. O wrapper brasileiro
fixado chama `F5TTS(ckpt_file=..., vocoder_name="vocos")` sem trocar o modelo.
No [módulo de atenção daquela versão](https://github.com/SWivid/F5-TTS/blob/33e865120c1ea08c39e9fb7b68e2ebd09b37ce58/src/f5_tts/model/modules.py),
RoPE é aplicado antes de reorganizar os 1.024 canais em 16 heads, girando 64
dimensões: corresponde ao primeiro head. Isso sustenta a opção atual
`pe_attn_head=1`, sem adotar a semântica diferente do v1 por coincidência de shapes.

O autor diz reutilizar o tokenizer F5 original. A comparação de `splitlines()`
confirmou igualdade das 2.545 entradas e seus IDs entre o vocabulário HF e o da
fonte; a diferença de hash decorre de CRLF/LF. Não foi inventado vocabulário
PT-BR. Espaço é ID zero e o frontend upstream usa esse ID para token desconhecido,
o que pode ocultar perda de caracteres: o probe registra os tokens OOV. Acentos
portugueses devem ser conservados. Não há dependência de eSpeak nem phonemização
portuguesa customizada nesse caminho; `convert_char_to_pinyin` preserva o texto
latino, com as regras de pontuação/segmentação do próprio upstream.

O card recomenda minúsculas e `num2words`. Seu exemplo substitui sequências de
dígitos isoladas; não constitui um normalizador correto de moeda, separador de
milhar, decimais, datas, horários ou siglas. O probe aplica somente NFC e,
quando declarado, minúsculas. Não converte silenciosamente os casos crus em
suas respostas esperadas. A coorte fixa conserva `numbers_raw` e
`numbers_spoken`, permitindo distinguir normalização de capacidade acústica.
Pronúncia de Aegis, intenção de leitura das siglas e acentuação ainda exigem
áudio e escuta; cobertura de caracteres não é avaliação de pronúncia.

Uma auditoria CPU do tokenizer real passou nos **nove textos** da coorte fixa:
zero tokens OOV após NFC/minúsculas e o frontend upstream, inclusive nos casos
crus de moeda, datas, horários e siglas. Recibo privado:
`f5/tokenizer-corpus-audit.json`. Os números continuaram escritos como dígitos;
o resultado comprova cobertura de tokens, sem demonstrar leitura correta deles.
O nome Aegis requer caso suplementar identificado, pois não está nesses nove
textos; acrescentá-lo não deve substituir a coorte principal.

Uma auditoria textual separada do normalizador atual do gateway também foi
executada em CPU. Datas/horas/unidades foram expandidas, mas a moeda virou
`R$ 1.247,oitenta e três`; SUS é soletrado, e o léxico mapeia Aegis para Égide.
Esse resultado não foi aplicado ao F5. Não se deve reutilizar esse normalizador
como solução já validada de PT-BR nem corrigir o benchmark alterando seu texto
sem declarar a variante. Recibo privado `gateway-normalization-audit.json`;
o serviço e seu normalizador permaneceram intactos.

## Ambiente realmente instalado

Ambiente privado: `/var/tmp/aegis-v2-evaluation-20261102/envs/f5/`.
Os caches experimentais permanecem sob
`/var/tmp/aegis-v2-evaluation-20261102/cache/`; o ambiente da produção não foi
alterado. Versões observadas no ambiente do roundtrip:

| Dependência | Versão |
| --- | --- |
| PyTorch / torchaudio | 2.6.0+cu118 / 2.6.0+cu118 |
| NumPy / SciPy | 1.26.4 / 1.17.1 |
| datasets / pyarrow | 2.21.0 / 17.0.0 |
| Vocos / x-transformers | 0.1.0 / 1.44.4 |
| torchdiffeq / librosa | 0.2.5 / 0.11.0 |
| accelerate / ema-pytorch | 1.15.0 / 0.8.3 |
| safetensors / soundfile | 0.8.0 / 0.14.0 |
| pypinyin / rjieba | 0.55.0 / 0.2.2 |

Houve falha real de importação com pyarrow 26 / NumPy 1.26; a combinação acima
foi instalada e passou no roundtrip. O primeiro diretório de tentativa foi
preservado; não se conta aquela tentativa como sucesso. `x-transformers` foi
fixado para compatibilidade da implementação, não assumido intercambiável com
qualquer versão futura. O runner evita Gradio, ASR e instalação automática do
conjunto completo de dependências upstream, mas o `model/__init__.py` oficial
importa o Trainer e exige parte das dependências de treinamento para importar
o modelo, mesmo em inferência.

O host tem driver 535.309.01, A2000 sm86 e indicação CUDA 12.2 no `nvidia-smi`.
A wheel selecionada usa CUDA 11.8; driver mais novo pode executar binários CUDA
anteriores pela [compatibilidade documentada pela NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
O roundtrip e os primeiros smokes são CPU com `CUDA_VISIBLE_DEVICES` vazio.
Isso verifica imports e execução acústica, **não** prova kernels, desempenho,
pico ou ausência de OOM na A2000. Não se alterou driver nem runtime de produção.

## Roundtrip obrigatório realizado

Foi usada uma referência privada tratada e curta de **5,04 s**, com transcrição
de origem vinculada pelo hash. Ela contém o efeito final já disponível na Aegis.
Essa referência é **24 kHz**; não há original 48 kHz correspondente disponível
neste ensaio. Não foi reamostrada para 48 kHz para simular um original. O ensaio
preserva A bit a bit e deixa explícita essa limitação: efeitos exclusivos da
banda acima de 12 kHz ainda não foram testados.

O caminho efetivamente executado é o do F5: `MelSpec(vocos)` com 24 kHz,
FFT/janela 1.024, hop 256, 100 mels, espectro de magnitude (`power=1`),
`center=True`, escala HTK e `log(clamp(mel,1e-5))` → `Vocos.decode(mel)`.
Não foi usado um codec diferente nem um bypass do frontend. A mesma extração
alimenta o cache da referência de inferência.

Artefatos em `f5/roundtrip-treated-02/`:

| Faixa | Duração / pico float | Observação |
| --- | --- | --- |
| A `A-original.wav` | 5,040000 s / original preservado | cópia byte a byte, hash conferido |
| B `B-input-24000.wav` | 5,040000 s / 0,489567 | mono float32 de entrada, ganho e duração preservados |
| C `C-vocos-24000.wav` | 5,034667 s / 0,422800 | saída neural float32, sem normalização posterior |

Em CPU, dois threads: carga Vocos **0,372 s**, leitura/conversão da referência
**1,427 ms**, mel **8,400 ms**, reconstrução Vocos **0,361 s**. C tem 128 samples
(5,333 ms) a menos que B, decorrentes do caminho mel/ISTFT; não foi preenchida
ou cortada manualmente para ocultar essa diferença. B e C têm zero samples fora
de `[-1,1]`. Isso demonstra uma reconstrução técnica finita, não preservação
perceptiva de robotização. Ataques, sibilantes, textura, formantes, ruído e
inteligibilidade precisam da escuta A/B. Os WAVs float32 preservam o resultado
neural, inclusive se algum ensaio posterior ultrapassar a faixa PCM16.

### Vocos e BigVGAN não são substitutos automáticos

O F5 suporta ambos, mas suas representações mel diferem. Vocos usa o caminho
torchaudio acima. O [extrator BigVGAN do F5](https://github.com/SWivid/F5-TTS/blob/45868abaabd74aabf03cd673d6dd00cc9d9c01ae/src/f5_tts/model/modules.py)
usa base mel librosa, padding reflexivo explícito e `center=False`, além da
fórmula própria de magnitude. Ter 100 canais, hop 256 e 24 kHz em ambos não
torna os tensores intercambiáveis. O [repositório oficial de pesos F5](https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_Base_bigvgan)
publica um checkpoint acústico **separado** `F5TTS_Base_bigvgan/model_1250000.pt`.
O card brasileiro demonstra Vocos e não fornece uma variante treinada para a
representação BigVGAN. Trocar só o decoder não é um ensaio válido desse peso.

Não foi baixado o segundo modelo acústico nem executada uma combinação sem
compatibilidade demonstrada. Um roundtrip independente de
`nvidia/bigvgan_v2_24khz_100band_256x`, com seu próprio mel e pesos/configs fixados,
é tecnicamente possível; seu sucesso provaria apenas a alternativa acústica,
não autorização para conectar o checkpoint PT-BR atual a ela.

Se a escuta aprovar a fala mas reprovar o efeito do Vocos, existe uma rota a
investigar: adaptar o vocoder separadamente, usando os WAVs tratados autorizados
e **o mesmo** extrator mel, com perdas espectrais/temporais e, se adequado,
adversariais. Só depois reavaliar ajuste do modelo acústico. A representação
mel perde fase e detalhes espectrais; adaptar o decoder não garante recuperar
efeitos que a condição não representa. Memória/duração desse treino de vocoder
na A2000 não foram medidas. Não foi iniciado ajuste de nenhum componente.

## Inferência CPU executada e medição A2000 pendente

O [probe](../../benchmarks/architecture_compare/f5_probe.py) reutiliza privacidade,
janela operacional e métricas de PCM do pacote existente. O plano não importa
Torch nem consulta GPU. Execução CPU não exige parar produção. CUDA exige
janela real para a candidata exata e GPU inicialmente vazia; cada chamada
confere exclusividade. O programa não para nem restaura o serviço operacional.

```bash
python3 benchmarks/architecture_compare/f5_probe.py

# Caminhos locais e recibo já preparados; escolher saída NOVA privada.
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  benchmarks/architecture_compare/f5_probe.py --execute --action roundtrip \
  --source-dir /var/tmp/aegis-v2-evaluation-20261102/f5/source \
  --vocab /var/tmp/aegis-v2-evaluation-20261102/f5/vocab.hf.txt \
  --vocoder-dir /var/tmp/aegis-v2-evaluation-20261102/f5/vocos \
  --reference-wav /CAMINHO_PRIVADO/referencia.wav \
  --reference-text-file /CAMINHO_PRIVADO/referencia.txt \
  --receipt /var/tmp/aegis-v2-evaluation-20261102/f5/probe-receipt.json \
  --output-dir /CAMINHO_PRIVADO/NOVO-ROUNDTRIP --device cpu
```

Para inferência, mudar para `--action infer`, acrescentar
`--checkpoint /var/tmp/aegis-v2-evaluation-20261102/f5/model_last.safetensors`
e selecionar `--case novel_short --nfe 8 16 32`. Sem `--case`, são usados todos
os textos da coorte fixa existente. A referência e sua transcrição precisam
corresponder integralmente. Não se corta a referência para caber numa duração
recomendada sem conferir a transcrição do trecho. O cache computa conversão
e mel uma vez, e todos os custos ficam separados das chamadas seguintes.

Cada frase é resolvida integralmente por `CFM.sample` e então decodificada.
Isso é **geração completa**, com um bloco de áudio; não streaming neural
incremental. O modo upstream que depois fatia o waveform pronto também não o
transforma em streaming neural. Euler usa 8, 16 ou 32 avaliações do campo,
`use_epss=False`, sway -1 e CFG 2; condicional/incondicional são empacotados no
mesmo forward com batch efetivo dois. A seed é preservada. A duração estimada
segue a razão de bytes UTF-8 entre texto e referência do upstream, com teto que
falha em vez de truncar silenciosamente.

O primeiro resultado é marcado `cold_generation=true` e preservado.
`--warmup` adiciona uma chamada NFE16 independente, também preservada e marcada;
nenhuma chamada lenta é descartada. O relógio de inferência começa depois da
carga e do preparo único da referência. Métricas incluem normalização/tokenização,
difusão, vocoder, PCM completo, proxy digital de PCM útil, reprodução estimada,
duração/RTF, blocos, falhas e memória quando houver CUDA. Salvar WAV/JSON acontece
depois do relógio de geração. Pico de carga CUDA e pico de inferência são
registrados separadamente. `load.json` preserva métricas de carga/referência
antes das gerações, inclusive se o watchdog encerrar uma janela parcial.
O cronômetro de imports/carga começa após validar arquivos/hashes; não inclui
essa métrica o custo de validação. O cache de disco não foi esvaziado para
simular uma carga fria física. O proxy digital não prova que os samples contêm
português correto ou fala inteligível.

O smoke CPU de `novel_short` com NFE 8/16/32 **concluiu as três tentativas**, sem
falhas, OOM, clipping ou tokens OOV observados. Todas estão preservadas em
`f5/infer-novel-short-01/`. Dados individuais sanitizados:
[f5-cpu-smoke-20261102.json](f5-cpu-smoke-20261102.json). Foi usado o Xeon
E5-2470 v2, dois threads, processo com prioridade nice 19 e GPU oculta, mantendo
o Qwen operacional. Não são medições da A2000 nem comparação de latência com
o Qwen acelerado.

Carga do modelo acústico: **3,908 s**; Vocos: **0,369 s**. Imports, carga e
preparo único da referência: **12,112 s** no total, após validar os hashes.
O cache do filesystem não foi esvaziado. Na chamada de inferência, a referência
de 5,04 s consumiu **1,366 ms** na leitura/conversão e **7,872 ms** no mel;
as três gerações reutilizaram o tensor mel `[1,100,473]`. Isso elimina essas
etapas repetidas, mas nesta CPU o custo dominante foi a difusão.

| NFE | Normalização/tokenização | Difusão | Vocos | Geração completa / PCM útil disponível | Reprodução estimada da primeira atividade | RTF |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0,174 ms | 210,458 s | 0,282 s | 210,741 s | 211,041 s | 51,585 |
| 16 | 0,119 ms | 427,740 s | 0,327 s | 428,067 s | 428,347 s | 104,781 |
| 32 | 0,481 ms | 847,943 s | 0,280 s | 848,224 s | 848,504 s | 207,627 |

Cada resultado tem **4,085333 s**, mono float32 a 24 kHz, um bloco de saída
e uma frase completa. O offset do proxy digital de atividade foi 300 ms em
NFE8 e 280 ms em NFE16/32. Como todo o waveform só fica disponível depois de
gerado, o PCM útil chega ao mesmo tempo que o restante; a reprodução estimada
soma o prefixo silencioso, sem confundir primeiro sample com fala. Zero gaps
nesse bloco único não valida streaming, continuidade em texto longo nem ausência
de omissão de palavras. Os picos float foram 0,9656 / 0,9923 / 0,9904, preservados
sem ajuste de loudness ou limitação.

A primeira geração NFE8 está marcada `cold_generation=true`; NFE16/32 usam o
mesmo processo já carregado. Não foi executado um warm-up separado, e nenhuma
chamada foi descartada como aquecimento. Portanto, a relação de tempo entre
NFE inclui essa diferença e só permite uma triagem do custo aproximado dos
passos. Uma seed, um texto e uma observação por configuração não justificam
percentis nem escolha acústica de NFE8. O ASR CPU transcreveu integralmente as três amostras NFE8/16/32.
ASR não substitui escuta de sotaque, timbre feminino, identidade ou textura
robótica. As três amostras devem ser ouvidas, sem selecionar apenas uma.

Faltam ainda a síntese do restante da coorte, caso suplementar de Aegis, textos
longos, números/siglas falados e a avaliação auditiva do proprietário. Memória,
carga, aquecimento, latência e RTF na A2000 continuam **não medidos**, sem
extrapolação dos resultados CPU.

## Fine-tuning: rota plausível, sem execução autorizada

A [receita oficial](https://github.com/SWivid/F5-TTS/blob/45868abaabd74aabf03cd673d6dd00cc9d9c01ae/src/f5_tts/train/README.md)
aceita CSV `audio_file|text`; o [Trainer fixado](https://github.com/SWivid/F5-TTS/blob/45868abaabd74aabf03cd673d6dd00cc9d9c01ae/src/f5_tts/model/trainer.py)
usa AdamW, acumulação de gradientes, EMA no dispositivo e checkpoints retomáveis
com modelo, otimizador, scheduler e contador. O DiT oferece
`checkpoint_activations`. Nenhum minibatch foi executado nesta investigação.

Para aproximadamente 337,1 milhões de parâmetros, uma conta explícita de estados
FP32 dá **1,256 GiB** para pesos; pesos + gradientes + dois momentos Adam dão
**5,023 GiB**; acrescentando EMA FP32, **6,279 GiB**. São estimativas aritméticas,
não VRAM medida. Faltam ativações, atenção, temporários, contexto CUDA e alocador.
AMP não divide automaticamente por dois todos esses estados. O armazenamento
persistente é menor que os 11.514 MiB visíveis da A2000, dando uma hipótese
plausível para batch um e clipes curtos, sem provar que o treino cabe. O otimizador
8-bit existe como opção upstream, mas não foi qualificado nesta máquina nem é
requisito inicial para justificar offloading.

Piloto proposto após autorização específica e corpus com direitos confirmados:
Base/Vocos coerentes; batch um, clipes curtos estratificados, AMP FP16,
`checkpoint_activations=True`, acumulação oito, vocoder inicialmente congelado e
validação PT-BR reservada. Começar com medição curta, preservar primeiras
iterações, medir pico e falhas; parar em OOM em vez de construir uma estrutura
grande de offloading. Salvar checkpoint retomável e snapshot de configuração,
seed, hashes, split e ambiente, mantendo quantidade limitada definida antes.

Um checkpoint completo Adam+EMA ocupa aproximadamente **5,39 GB**, coerente
com os `.pt` publicados. Duas versões podem consumir ~10,8 GB além do peso
inicial e ambiente; isso precisa caber no orçamento restante real. Não criar
retenção ilimitada. Avaliar tanto pesos normais quanto EMA: o upstream alerta
que EMA pode conservar demais a voz pré-treinada no início de um ajuste curto.

**Não há estimativa honesta em horas baseada na A2000 ainda.** Ela exige uma
medição de forward/backward/optimizer com a configuração de treino autorizada.
Para planejar sem inventar um benchmark: `tempo ≈ N_microsteps × t_microstep +
validação + checkpoints`; com acumulação oito, 100 updates requerem 800
microsteps. Registrar mediana e cauda após o primeiro ciclo, depois estimar
o piloto e a escala do corpus. Tempos do A100/T4 relatados pelo autor e RTF de
inferência CPU não são substitutos dessa medição.

A importação futura mantém WAVs/hash, texto original e texto revisado, IDs,
grupos/splits e proveniência. Ver [importação v2](dataset-import-v2.md).
Áudios de ElevenLabs não entram em treino sem evidência de que os termos
aplicáveis permitem esse uso específico. Uma boa clonagem por referência não
aprova automaticamente o dataset, o fine-tuning ou a promoção da nova voz.
