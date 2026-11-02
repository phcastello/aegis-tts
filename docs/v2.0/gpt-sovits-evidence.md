# GPT-SoVITS v2Pro: evidências e impedimentos para o ensaio PT-BR

Consulta em 02/11/2026; somente fontes, metadados e código. Nenhum peso foi
baixado e nenhuma inferência ou treinamento foi executado. Os campos de VRAM,
tempo até PCM útil, RTF, estabilidade e qualidade auditiva na A2000 continuam
**não medidos**.

## Identificação e português brasileiro

O checkpoint encontrado é
[`MatheusMarquesEiras/gpt-sovits-ptbr-base`](https://huggingface.co/MatheusMarquesEiras/gpt-sovits-ptbr-base/blob/ba75b17c00876e6ab5324251f7d34a099ca8bf3c/README.md),
revisão `ba75b17c00876e6ab5324251f7d34a099ca8bf3c`. O autor declara v2Pro,
com GPT derivado de `s1v3` e SoVITS de `s2Gv2Pro`, e classifica o trabalho
como experimental. O frontend usa eSpeak-ng `pt-br`, 88 símbolos com prefixo
`P`, IDs 732–819 e embeddings de 820 linhas. A primeira fase utilizou um
falante do TTS-Portuguese Corpus; não há avaliação perceptual publicada no card.
O relato de treino numa RTX 4070 de 12 GB é evidência externa de execução,
sem medição de pico e sem demonstrar viabilidade na A2000.

**Impedimento atual:** o card exige `symbols2.py` modificado e
`text/portuguese.py`, mas não identifica uma revisão de código para obtê-los.
A [árvore do checkpoint](https://huggingface.co/MatheusMarquesEiras/gpt-sovits-ptbr-base/tree/ba75b17c00876e6ab5324251f7d34a099ca8bf3c)
contém pesos e metadados, sem esses arquivos. Não foi encontrado um frontend
reproduzível associado ao checkpoint nas fontes consultadas.

O upstream inspecionado foi fixado em
[`04a44eb977586645f33594b72af13a035a2056c6`](https://github.com/RVC-Boss/GPT-SoVITS/tree/04a44eb977586645f33594b72af13a035a2056c6).
Seu [cleaner](https://github.com/RVC-Boss/GPT-SoVITS/blob/04a44eb977586645f33594b72af13a035a2056c6/GPT_SoVITS/text/cleaner.py)
só mapeia zh/ja/en/ko/yue e converte idioma desconhecido para inglês com texto
vazio. A
[configuração de inferência](https://github.com/RVC-Boss/GPT-SoVITS/blob/04a44eb977586645f33594b72af13a035a2056c6/GPT_SoVITS/TTS_infer_pack/TTS.py)
também não aceita `pt`/`pt-br`. Portanto, baixar os pesos e selecionar português
no código original não funciona. É preciso obter o frontend exato, sua licença,
a ordem dos símbolos e os pontos de integração na API antes de baixar pesos.
Recriar arbitrariamente a lista de 88 fonemas pode produzir IDs semanticamente
errados mesmo com shapes corretos.

## Licenças e artefatos

| Componente | Licença declarada e fonte |
| --- | --- |
| Código GPT-SoVITS | [MIT](https://github.com/RVC-Boss/GPT-SoVITS/blob/04a44eb977586645f33594b72af13a035a2056c6/LICENSE). |
| GPT/SoVITS adaptados PT-BR | CC BY 4.0 no card acima; conservar atribuição ao autor e ao corpus. A licença do código customizado do frontend ainda precisa ser localizada. |
| Pesos base e ERes2Net distribuídos pelo upstream | [MIT no repositório de pesos](https://huggingface.co/lj1995/GPT-SoVITS/blob/8f8857563c040d542de108a2c18b0425abd6c9c5/README.md); a distribuição inclui o [checkpoint SV](https://huggingface.co/lj1995/GPT-SoVITS/blob/336b2ec4e8d4ac74740798dd40af44e74659ecaf/sv/pretrained_eres2netv2w24s4ep4.ckpt). Isso não relicencia componentes com avisos próprios. |
| CNHuBERT | [MIT na origem](https://huggingface.co/TencentGameMate/chinese-hubert-base). |
| Chinese RoBERTa | [Apache 2.0 na origem](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large). |
| eSpeak-ng | [GPLv3](https://github.com/espeak-ng/espeak-ng/blob/master/COPYING), ferramenta externa usada pelo frontend descrito. Não é Piper. |

Para um primeiro par coerente da fase 1, os metadados LFS fixam:

| Arquivo | Bytes | SHA-256 publicado |
| --- | ---: | --- |
| `gpt/ptbr_base-e20.ckpt` | 155403680 | `4f5094799d2d6169ebb1b6b6cb806d1c7a06399e8ad263a42cb72c98c61332b2` |
| `sovits/ptbr_base_e6_s3210.pth` | 134980339 | `1a05d5584d4434d2d639b8cd716eafb7e62b03f2d8e00561c27551a65518f08e` |

Fonte: [API da árvore fixada](https://huggingface.co/api/models/MatheusMarquesEiras/gpt-sovits-ptbr-base/tree/ba75b17c00876e6ab5324251f7d34a099ca8bf3c?recursive=true&expand=false).
Não selecionar automaticamente o maior número de época ou misturar fase 2
sem protocolo de comparação. O
[`retomada/MANIFESTO.json`](https://huggingface.co/MatheusMarquesEiras/gpt-sovits-ptbr-base/blob/ba75b17c00876e6ab5324251f7d34a099ca8bf3c/retomada/MANIFESTO.json)
distingue checkpoints com otimizadores dos arquivos de inferência: `retomada/`
não é necessário nesta etapa.

O runtime original carrega também CNHuBERT, RoBERTa e ERes2Net. Os cinco blobs
principais, usando os arquivos espelhados em
[`lj1995/GPT-SoVITS@8f885756...`](https://huggingface.co/api/models/lj1995/GPT-SoVITS/tree/8f8857563c040d542de108a2c18b0425abd6c9c5?recursive=true&expand=false),
somam **1.237.949.278 bytes**, além de configs, tokenizer, código e ambiente.
Isso é tamanho em disco, não estimativa de VRAM. O repositório PT-BR inteiro tem
várias épocas e estados de treino; não há motivo para baixar os 6,69 GB completos.

## Áudio, codec e banda

A [receita v2Pro](https://github.com/RVC-Boss/GPT-SoVITS/blob/04a44eb977586645f33594b72af13a035a2056c6/GPT_SoVITS/configs/s2v2Pro.json)
define saída **32.000 Hz**, FFT 2048 e hop 640; a API entrega PCM mono
**int16**. O cálculo do modelo pode usar FP16 ou FP32, o que não é uma
profundidade acústica nativa. Confirmar `config.data.sampling_rate=32000` nos
pesos PT-BR e no WAV efetivamente recebido durante o futuro ensaio.

A arquitetura contém GPT autorregressivo para tokens semânticos, CNHuBERT,
quantizador residual, fluxo e decoder de forma de onda do SoVITS, com guia de
speaker embedding ERes2Net. Ela contém gargalos de representação mesmo sem o
codec Qwen. O
[modelo SoVITS](https://github.com/RVC-Boss/GPT-SoVITS/blob/04a44eb977586645f33594b72af13a035a2056c6/GPT_SoVITS/module/models.py)
usa apenas `refer[:, :704]` no encoder de timbre v2; para FFT 2048 a 32 kHz,
isso corresponde aproximadamente a 11 kHz na referência (inferência a partir
dos bins, não medição de banda sintetizada). O caminho CNHuBERT recebe 16 kHz.
Esses limites de condicionamento não são um corte direto da saída em 11 kHz,
mas tornam obrigatória a reconstrução e a escuta dos efeitos. Não há evidência
de que o decoder preserve aspereza, ataques, sibilantes e harmônicos tratados.
A saída 32 kHz tem Nyquist de 16 kHz; reamostrar para 48 kHz não adiciona banda.

## Streaming e runtime que devem ser medidos

Na [API v2](https://github.com/RVC-Boss/GPT-SoVITS/blob/04a44eb977586645f33594b72af13a035a2056c6/api_v2.py),
`streaming_mode=1` ou `true` significa retorno por fragmento/frase completo.
Os inteiros `2` e `3` acionam outro caminho: o
[gerador AR](https://github.com/RVC-Boss/GPT-SoVITS/blob/04a44eb977586645f33594b72af13a035a2056c6/GPT_SoVITS/AR/models/t2s_model.py)
emite tokens antes do EOS, e `TTS.run` chama `decode_streaming` com prefixos,
overlap e descarga de cauda. Existe, portanto, implementação incremental no
código para v2Pro. Ela ainda não foi validada com o frontend/pesos PT-BR, nem
medida na A2000. Modo 3 troca o critério de corte e pode afetar qualidade.

O novo [`gpt_streaming=true`](https://github.com/RVC-Boss/GPT-SoVITS/blob/04a44eb977586645f33594b72af13a035a2056c6/GPT_SoVITS/TTS_infer_pack/streaming.py)
com aceleração AR separada **rejeita v2Pro** nessa revisão: só aceita
v2ProPlus/v5turbo/v5dev. Não tratar esse recurso como aceleração disponível para
o checkpoint PT-BR. Também não presumir integração nativa com vLLM.

Primeiro runtime a validar: PyTorch CUDA FP16, batch 1, sem CUDA Graph ou
FlashAttention adicionais; depois medir cada aceleração que de fato carregar.
O [README testado pelo upstream](https://github.com/RVC-Boss/GPT-SoVITS/blob/04a44eb977586645f33594b72af13a035a2056c6/README.md)
inclui PyTorch 2.7.1/CUDA 11.8. O
[`install.sh`](https://github.com/RVC-Boss/GPT-SoVITS/blob/04a44eb977586645f33594b72af13a035a2056c6/install.sh)
atual instala conjuntos maiores e CUDA 12.6/12.8; não deve ser executado
cegamente no host. CUDA 11.8 é uma primeira hipótese conservadora para um
ambiente isolado, sujeita à compatibilidade das dependências escolhidas.

O driver local 535 não autoriza supor suporte a qualquer kernel recente.
A [NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
permite compatibilidade menor dentro de CUDA 12 com driver mínimo 525,
mas há restrições de funcionalidades; extensões/PTX precisam ser conferidas.
O ensaio deve registrar `torch.version.cuda`, cuDNN, capability `(8,6)`,
dependências, backend realmente usado e falhas de fallback. Nenhuma troca
de driver ou imagem de produção é parte desta preparação.

Após resolver o frontend e durante janela autorizada, o comando de servidor
original pode ser preparado em checkout/ambiente próprio:

```bash
python api_v2.py -a 127.0.0.1 -p 19882 -c /PRIVADO/gpt-sovits-ptbr.yaml
```

O YAML deve fixar `custom.version: v2Pro`, `device: cuda`, `is_half: true`,
os dois pesos PT-BR e caminhos locais de CNHuBERT/RoBERTa; o SV fica no caminho
esperado pelo checkout isolado. A API escreve configuração durante o load;
usar cópia privada gravável. Somente HTTP loopback. Comparar modo 0 e modo 2,
batch 1, `text_split_method: cut0`, PCM raw e mesmo texto sem pontuação interna
para provar PCM anterior ao fim da geração, registrando tempo até fala digital
útil, EOF, RTF, gaps e cauda. O código exato do frontend determinará se a chave
de idioma é `pt`, `pt-br` ou outra; esse valor continua pendente.

A receita de medição preparada está em
[`candidate.gptsovits.unverified.json`](../../benchmarks/architecture_compare/candidate.gptsovits.unverified.json),
com `runtime_verified=false` e idioma explicitamente não resolvido. Ela aponta
para `/tts` real da API v2 e fixa `media_type=raw`, `streaming_mode=2`,
`gpt_streaming=false`, batch 1 e `cut0`. Os hashes são metadados remotos, ainda
não verificação dos arquivos locais. Não habilitar a receita antes de resolver
os campos e os impedimentos indicados. Exemplo do YAML privado, que também
depende desse frontend:

```yaml
custom:
  version: v2Pro
  device: cuda
  is_half: true
  t2s_weights_path: /PRIVADO/pesos/gpt/ptbr_base-e20.ckpt
  vits_weights_path: /PRIVADO/pesos/sovits/ptbr_base_e6_s3210.pth
  bert_base_path: /PRIVADO/pesos/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: /PRIVADO/pesos/chinese-hubert-base
```

Manter os pesos SV em
`GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt` dentro
do checkout isolado, conforme `SV` upstream; esse caminho é independente dos
quatro campos de pesos configuráveis acima. Preservar também os configs e
tokenizer correspondentes aos dois diretórios Hugging Face.

## Fine-tuning: caminho existe; memória continua pendente

Há treinamento separado de GPT e SoVITS no upstream, incluindo decoder e
discriminador do SoVITS. O
[`s2_train.py`](https://github.com/RVC-Boss/GPT-SoVITS/blob/04a44eb977586645f33594b72af13a035a2056c6/GPT_SoVITS/s2_train.py)
suporta v2Pro e preserva estados do otimizador; a configuração oferece batch,
FP16 e checkpoint de gradientes. O card PT-BR relata batch 6 de GPT na 4070,
sem pico de VRAM. Isso torna o fine-tuning uma hipótese plausível, sem comprovar
12 GB operacionais na A2000. Não transferir números de inferência para treino:
otimizadores, gradientes, ativações e discriminadores mudam a memória.

Antes de qualquer treino futuro, exigir A/B de reconstrução sem atualização de
pesos e de fala inédita, identificação vocal reconhecível e efeitos aprovados
pelo proprietário. A execução de treinamento permanece fora desta etapa,
inclusive minibatches de benchmark e extração massiva de corpus. A candidata
só avança depois de obter frontend compatível e superar esses testes reais.
