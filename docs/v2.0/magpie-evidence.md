# MagpieTTS v2607 e NanoCodec — evidência de 02/11/2026

Magpie permanece candidata secundária. **O roundtrip do áudio robótico tratado
foi executado em CPU**, com encoder, FSQ e decoder oficiais e todos os tensores
carregados estritamente. O frontend brasileiro também foi executado isoladamente.
Ainda não houve síntese Magpie, carregamento CUDA, teste de fine-tuning ou avaliação
auditiva do proprietário. Nenhum resultado daqui prova identidade da futura Aegis.

## Artefatos fixados e licenças

O [card v2607](https://huggingface.co/nvidia/magpie_tts_multilingual_357m/blob/5023df68bd3f5b5ce6d666a50979bc501af145cc/README.md)
identifica 364 milhões de parâmetros, português entre os idiomas e cinco vozes
fixas. A clonagem zero-shot foi removida. Na API Hub, `v2607` é uma **branch**,
resolvida a `5023df68bd3f5b5ce6d666a50979bc501af145cc`; não se usa uma referência
mutável ao executar. O artefato é o mesmo `.nemo` servido na main consultada.

| Componente | Revisão | Arquivo | Bytes | SHA-256 do manifesto Hub |
| --- | --- | --- | ---: | --- |
| Magpie acústico | `5023df68bd3f5b5ce6d666a50979bc501af145cc` | `magpie_tts_multilingual_357m.nemo` | 1.470.208.000 | `ec675fa8c02b9c1d5382c5c2b5a6acec6492c1e8344866c07cf3892185d18953` |
| NanoCodec 22 kHz / 1,89 kbps / 21,5 fps | `fc00890b604aa2de298d2641ffc6c5f6caf8c4d7` | `nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo` | 425.021.440 | `28c2518de3e3d5a2c7d9bca40a7ebc0644eb76c60b890970365325bdd8e9f099` |

**Somente o codec foi baixado integralmente e teve SHA-256 local conferido.**
O checkpoint Magpie teve manifesto e cabeçalhos/config/dicionário inspecionados
por HTTP Range, sem download do tensor acústico de 1,44 GB e sem desserialização.
O tar contém múltiplas versões de `model_config.yaml`; deve-se usar a **última**,
como faz `tarfile.getmember`. Ela inclui o mapeamento de linguagem e a opção
`locale_specific_punct: false` necessários à compatibilidade do tokenizer.

A [licença Magpie](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
permite uso comercial e derivados, com condições: redistribuição requer acordo
e atribuição, aplicação dos termos de uso e preservação de salvaguardas adequadas.
O card NanoCodec aponta a [versão de junho de 2024](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf),
também comercial e com obrigações de redistribuição/uso. Não confundir essas
licenças com a Apache-2.0 do código NeMo ou assumir que são domínio público.
Nenhum teste tentou reativar clonagem zero-shot.

## Frontend PT-BR: resultado executado

Fonte [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech/tree/e5d79d5a49cbec49d79440b0070aac0c32a8dbb8)
fixada em `e5d79d5a49cbec49d79440b0070aac0c32a8dbb8`. O config do checkpoint
define `portuguese_Brazilian_phoneme`: `IPATokenizer(locale='pt-BR')`,
`IpaG2p(locale='pt-BR')`, dicionário embarcado
`05adc40366e149b69319acd4b28a4919_pt_br_prondict-v1.0.dict`, caracteres com prefixo
`#` e caixa alta, acentos/IPA preservados. O dicionário tem 3.580.284 bytes e hash
`6492abc404db16bbad83dd9d7f9a60eb617699f0a3ff54b3be6184e14507b745`.
O canal de condicionamento textual adicional usa o tokenizer de `google/byt5-small`;
isso não substitui o G2P de fala. Não foi baixado o modelo ByT5.

[magpie_frontend_probe.py](../../benchmarks/architecture_compare/magpie_frontend_probe.py)
executa as definições Python oficiais, omitindo apenas inicialização/imports do
pacote NeMo, sem Torch. Hashes de todas as fontes e do dicionário são conferidos.
Para este diagnóstico fixou-se explicitamente probabilidade fonêmica 1,0; o
config original usa 0,8. Não se presume que o runtime completo aplique esse
override sozinho. A pontuação específica do locale permanece desativada,
conforme o checkpoint. Obteve **111 tokens**; SHA-256 da lista JSON UTF-8:
`d338ced2ca52313c975f8a83825eaf8c1e514a272d46756119aaa71c24e2e334`.

A coorte fixa de nove textos não foi alterada. Sete textos não perderam símbolos.
Em `numbers_raw` o tokenizer **descartou dígitos, moeda e percentuais**; em
`dates_units` descartou dígitos e sinais prefixados. `numbers_spoken` passou sem
símbolos perdidos. CPU/CPF/URL/HTTPS e kHz não constam do dicionário, mas possuem
fallback grafêmico; isso exige audição, não prova pronúncia correta. O suplemento
Aegis também caiu em grafemas `#A #E #G #I #S`, sem pronúncia fonética definida.
Não se inventou um IPA para o nome.

Há um problema concreto de integração no pin inspecionado:

1. O mapa do checkpoint contém **`pt-BR`**, mas `get_tokenizer_for_language`
   exige correspondência exata e faz fallback para inglês quando recebe `pt`.
2. `do_tts(..., language='pt-BR', apply_TN=True)` repassa o locale ao normalizador.
   O [TextProcessing consultado](https://github.com/NVIDIA/NeMo-text-processing/blob/ddadfb2a38d2bc6b8cc6232c4f915eb60f500688/nemo_text_processing/text_normalization/normalize.py)
   (`ddadfb2a38d2bc6b8cc6232c4f915eb60f500688`) aceita `pt`, mas não `pt-BR`.
   O código Magpie captura a exceção e continua com texto cru.

A rota proposta é normalização explícita externa `Normalizer(lang='pt')`, seguida
de `do_tts(language='pt-BR', apply_TN=False)`, guardando texto original, normalizado
e tempos. Essa normalização **ainda não foi executada**: o probe mede G2P sem TN,
e não se instalou Pynini/TextProcessing apenas para contornar silenciosamente
esse problema. A hipótese deve ser testada com moeda, data, hora, siglas e intenção
do proprietário antes de publicar um adaptador. Acertar o locale não demonstra
sotaque ou inteligibilidade por si só.

Resultados privados: `magpie/frontend-corpus.json` e
`magpie/frontend-aegis-supplement.json`, sob a raiz do experimento.

## Roundtrip acústico executado

O [NanoCodec correspondente](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)
opera a **22.050 Hz**, 1.024 samples/frame, oito grupos FSQ `[8,7,6,6]`, 2.016
valores por grupo e representação de dimensão 32. A configuração real usa
`HiFiGANEncoder` não causal e `CausalHiFiGANDecoder`. Taxa menor que 32 kHz não
elimina o modelo; interessa a retenção perceptiva do efeito.

O ensaio usou o áudio **já tratado** de produção, 5,04 s, FLOAT mono 24 kHz;
não uma voz natural apresentada como referência robótica. Seu original foi
copiado bit a bit, hash
`338968eebd4e7d5aa974d1b1060b6925202fe13798896fb7cea23ec29a63869b`.
Não existe original 48 kHz desse trecho identificado nesta sessão; não foi
fabricado por upsampling. Todos os originais existentes permanecem intactos.

| Faixa | Conteúdo |
| --- | --- |
| A | Original tratado, FLOAT 24 kHz, sem alteração |
| B | Resample para FLOAT 22.050 Hz, `scipy.resample_poly`, Kaiser 10, sem ganho/trim |
| C | B → pad final para frames completos → encoder → FSQ encode/decode → decoder, FLOAT 22.050 Hz |

Preservaram-se também B/C PCM16 para escuta offline. Não houve limiter,
normalização de loudness, denoiser nem corte de silêncio. B tem 111.132 samples;
C tem 111.616: os 484 samples finais de padding (~21,95 ms) estão preservados.
Tokens: `[8,1,109]`. Picos B/C: 0,488 / 0,476; RMS: 0,07281 / 0,07081;
zero NaN/Inf ou samples em escala cheia. Essas medidas não decidem se sibilantes,
ataques, formantes, textura robótica ou inteligibilidade foram preservados.

O [harness específico](../../benchmarks/architecture_compare/magpie_codec_probe.py)
verifica hashes antes de usar `torch.load(weights_only=True, map_location='cpu')`.
Seleciona via AST as definições oficiais e suas dependências; desativa somente
decorators de metadados NeMo/TorchScript e usa `nn.Module` no lugar da base
NeMo. Fórmulas numéricas, pesos, padding, convoluções e quantização permanecem
as do fonte fixado. Carga **strict**: encoder 291 chaves, FSQ 16, decoder 387;
nenhuma chave ausente/inesperada. Encoder 31.019.224 parâmetros e decoder
31.581.690. Discriminadores e losses de treino não são necessários ao roundtrip.
Isso valida o caminho numérico, **não o carregamento completo do NeMo**.

CPU Xeon E5-2470 v2, Torch 2.6.0+cu118, dois threads, prioridade nice 19,
`CUDA_VISIBLE_DEVICES=''`: carga de checkpoint/componentes **7,50 s**,
resample **0,0058 s**, encoder **4,83 s**, FSQ encode **0,0072 s**, FSQ decode
**0,0023 s**, decoder **6,50 s**; RTF do roundtrip CPU **2,249**. Uma execução,
sem warm-up descartado. Não são latências de síntese Magpie ou resultados A2000.

Artefatos: `/var/tmp/aegis-v2-evaluation-20261102/magpie/roundtrip-treated-v1/`
contém A/B/C, floats, PCM16, hashes e `receipt.json` (0700/0600).
O conjunto exige revisão A↔B, B↔C e A↔C pelo proprietário. Se C remover a
robotização desejada, o codec **congelado no treino Magpie** limita a vantagem
de adaptar apenas o locutor; não se presume que SFT acústico corrija essa perda.

## Runtime e treinamento: possibilidade ainda não demonstrada

O ambiente compartilhado privado F5 (Python 3.12, Torch 2.6.0+cu118) executou o
codec em CPU. Não foi instalado o stack NeMo completo. O pyproject do pin Speech
exige `torch>=2.7.0`; seu extra TTS inclui Lightning, lhotse, normalizadores,
tokenizers e vários idiomas. Não se deve inferir compatibilidade completa a partir
do harness. O driver 535.309.01 aceita a wheel cu118 instalada; ainda falta validar
uma combinação NeMo/PyTorch/CUDA pinada no teste exclusivo, sem trocar o driver.
`AudioCodecModel` integral também pode buscar auxiliares de treino automaticamente;
um runtime offline deve fixar codec local e desabilitar losses auxiliares, sem
introduzir downloads de speaker encoder/WavLM no meio da janela.

O card atual menciona runtime C++/GGUF menor, mas não foi adotado para este ensaio:
decoder isolado não realiza encode→decode, e sua compatibilidade de tokenizer/FT
não foi validada. Não foi criada infraestrutura de offloading.

A [receita oficial de fine-tuning](https://docs.nvidia.com/nemo/speech/nightly/tts/magpietts-finetuning.html)
prevê novo locutor, LR baixo e cache opcional de tokens. O manifest preserva
`audio_filepath`, `text`, `duration`, `context_audio_filepath`, `context_text`;
dataset metadata deve selecionar `portuguese_Brazilian_phoneme`. O codec permanece
congelado. O exemplo recomenda FP32, batch 16 e oito GPUs; não fornece pico de VRAM
para uma A2000. Há diferença entre essa receita genérica e o artefato v2607 com
embeddings de contexto fixos: o código omite pesos do context encoder nesses
checkpoints e não torna o embedding fixo treinável automaticamente. O piloto
deve verificar o caminho de especialização de locutor, sem anunciar clonagem.

Estimativa contábil, **não medição**: 364M parâmetros FP32 + gradientes + dois
estados Adam exigem ~5,42 GiB, antes de ativações, codec, buffers e framework.
Isso não prova que 12 GB bastem nem que não bastem. Um piloto autorizado deveria
começar com batch 1, clipes curtos e cache de códigos, medir pico real e tempo
por update, então projetar `updates × segundos/update + validação/checkpoint`.
Não há estimativa honesta de horas na A2000 sem esse passo; o tempo CPU do codec
não é throughput de treinamento. Não houve treino nem uso de outputs ElevenLabs
como dataset de treino nesta sessão.

## Reprodução e decisão

Fontes/configs/metadados pequenos e o codec verificado estão em
`/var/tmp/aegis-v2-evaluation-20261102/magpie/`; o recibo
`artifact-receipt.json` registra os bytes/hashes. Uso total medido: **415 MiB**
incluindo roundtrip. Ambiente e cache são compartilhados com F5 e devem ser
contados uma única vez no orçamento global. Não se baixou o repo HF completo.

```bash
python3 benchmarks/architecture_compare/magpie_frontend_probe.py \
  --artifact-dir /var/tmp/aegis-v2-evaluation-20261102/magpie

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  nice -n 19 /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  benchmarks/architecture_compare/magpie_codec_probe.py --execute \
  --artifact-dir /var/tmp/aegis-v2-evaluation-20261102/magpie \
  --audio /CAMINHO_PRIVADO/treated-v1.wav --output /NOVO_DIRETORIO_PRIVADO

python3 -m pytest -q tests/test_magpie_codec_probe.py
```

Três testes CPU passaram: plano sem dependências, recusa de fonte/pesos alterados
e preservação das expressões numéricas ao remover decorators. O roundtrip real
é uma evidência adicional independente desses testes. Faltam audição e runtime
de síntese antes de disputar a prioridade do F5. A qualidade de uma voz padrão
Magpie futura deverá avaliar pronúncia, nunca identidade zero-shot inexistente.
