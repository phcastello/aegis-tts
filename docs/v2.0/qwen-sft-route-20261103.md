# Qwen0.6B Base: especialização robótica sem WORLD duplicado

**Há uma rota oficial de SFT de locutor único para0.6B Base.** A hipótese é
ensinar o caráter robótico diretamente nos códigos acústicos da Aegis e avaliar
uma saída experimental sem WORLD/reserva herdados. Ela só avança se reconstrução
do codec, clareza PT-BR, identidade e TTFA passarem localmente. Não altera o
perfil protegido nem as prioridades: VITS idioma em curso, FastPitch PT em
seguida e VoxCPM2 condicionado ao piloto. Nesta pesquisa não houve treino,
carregamento de modelo, uso de GPU ou mudança de configuração de produção.

## O que a receita realmente suporta

O [guia oficial](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/README.md)
nomeia explicitamente `Qwen3-TTS-12Hz-1.7B/0.6B-Base` e locutor único.
O exemplo usa1.7B por padrão, mas o caminho inicial é argumento; não há no
treinador uma restrição a1.7B. A Base local tem `tts_model_size=0b6`, hidden1024,
speaker encoder1024 e vocabulário acústico3072: a posição3000 usada na exportação
existe. A saída passa a `tts_model_type=custom_voice`, com speaker experimental
novo; não é necessário partir de pesos CustomVoice. No0.6B o argumento `instruct`
não é suportado, mas um locutor especializado continua previsto na API.

O [treinador](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py)
ajusta o talker e o preditor dos códigos residuais; loss combina CE do primeiro
codebook e0,3 da loss residual. O speaker encoder produz um vetor com `detach`,
sem aprender. O primeiro vetor de referência é retido e inserido na linha3000
do embedding acústico ao exportar; esse vetor não é um embedding de locutor
aprendido separadamente. A identidade desejada deve vir também da adaptação
do gerador aos códigos de áudio, não apenas desse vetor. Usar a mesma referência
robótica curta de **treino** em todas as linhas evita selecionar um vetor
arbitrário pela primeira amostra embaralhada. Não usar uma âncora de teste ou
substituir a referência privada protegida.

O codec/decoder não participa dessa otimização; a ausência de nomes
`speech_tokenizer.*` no safetensors principal também confirma os componentes
separados. Se a reconstrução já remover as pistas robóticas necessárias, esse
SFT não conserta a limitação do codec. Não foi encontrada uma receita oficial
de treino conjunto de codec/vocoder na pasta `finetuning` inspecionada.

O [dataset oficial](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/dataset.py)
aceita `audio`, `text`, `ref_audio` e `audio_codes[T,16]`. A referência exige
24kHz; sua mel tem128 bandas. O campo `language` é lido, mas não entra no retorno
nem no collator: acrescentar `"language":"Portuguese"` ao JSON não muda a
receita. A primeira linha reproduzível deve manter o prefixo Auto oficial nos
dois sistemas. Uma experiência com prefixo explícito português exige adaptar e
testar os offsets de texto, speaker, BOS/EOS e labels; não basta trocar um rótulo.

## Insumos, versão e licenças

Código Qwen Apache-2.0, pin Git **`022e286b98fbec7e1e916cb940cdf532cd9f488e`**,
o mesmo consultado em02/11. A Base e o tokenizer têm declaração Apache-2.0 em
seus [cards Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/5d83992436eae1d760afd27aff78a71d676296fc/README.md)
e [codec](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/blob/7dd38ad4e9bad454aae9cd937d0cd577604fe229/README.md).
Isso não muda a privacidade/direitos dos áudios Aegis. Manter pesos ajustados,
transcrições e referências locais até definir permissão de redistribuição.

Todos os arquivos necessários da Base local já existem em
`/home/phc/aegis-tts/models/Qwen3-TTS-12Hz-0.6B-Base`, **2.516.107.412 bytes**,
inventariados antes desta pesquisa sem carregamento. Não baixar outra Base:

| Arquivo | Bytes | SHA256 |
| --- | --- | --- |
| `model.safetensors` | 1.829.344.272 | `180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6` |
| `speech_tokenizer/model.safetensors` | 682.293.092 | `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258` |
| `config.json` | 4.494 | `2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011` |

O código exige `transformers==4.57.3` e `accelerate==1.12.0`. O ambiente F5
tem Torch2.6.0+cu118, Transformers4.49.0 e Accelerate1.15.0; não é um ambiente
Qwen validado. Não modificar esse ambiente compartilhado durante VITS/F5.
Preparar um overlay isolado ou ambiente dimensionado depois das linhas ativas.
O script fixa FlashAttention2, ausente no ambiente. O modelo declara suporte
a SDPA e gradient checkpointing no
[código oficial](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py);
um piloto pode testar SDPA PyTorch sem compilar um pacote grande, registrando
essa diferença. Nenhuma instalação foi feita nesta pesquisa.

## Primeiro gate obrigatório: reconstrução reservada pelo codec

Reservar um trecho real tratado da voz robótica que não participe do treino.
Preservar WAV original e criar controle mono24k sem normalizar ganho, pitch ou
aplicar WORLD. Comparar original, controle reamostrado e reconstrução12Hz. O
[wrapper oficial](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_tokenizer.py)
oferece `encode` e `decode` diretamente; não precisa do gerador inteiro:

```python
# Protocolo para futura janela exclusiva; NÃO executado nesta pesquisa.
codec = Qwen3TTSTokenizer.from_pretrained(
    "/home/phc/aegis-tts/models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer",
    device_map="cuda:0", local_files_only=True,
)
encoded = codec.encode(controle_24k_f32, sr=24000)
reconstructed, sr = codec.decode(encoded)
```

Conferir número de codebooks, duração/padding, valores finitos, clipping, hashes,
energia por banda, ataques e sibilantes; preservar poucos pares para escuta.
Não inferir identidade pelo espectro ou ASR. Se o decoder final usar BF16,
comparar essa precisão com a reconstrução FP32 antes de escolher. Não aplicar
WORLD depois e apresentar o resultado como aprendizado dos pesos. Esse gate
ainda está **não medido**, sem inviabilizar a investigação atual de VITS.

## Dados e armazenamento

Qwen já tem português na base; essa linha não requer pré-treino no corpus
público de8h usado pelas arquiteturas sem idioma comprovado. Usar as partições
fixas Aegis e verificar quais WAVs realmente expressam a robotização desejada.
O [preparador oficial](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/prepare_data.py)
extrai códigos em batches constantes de32 e escreve tudo somente ao final.
Adaptar para microbatch1, saída por partes/atômica, hashes e retomada antes de
processar o conjunto. A referência fixa pode ter mel/vetor calculados uma vez.

Para4h:12,5 frames/s ×16 codebooks = **2.880.000 inteiros**; cerca de23,04MB
em int64 ou5,76MB em uint16, mais metadados. JSON deve ficar na ordem de20–30MB,
medir antes de projetar; listas Python consomem mais RAM, mas não oito horas
de novos WAVs/mels. Esses são cálculos, não uma extração executada. Não duplicar
áudio original para outro ambiente.

O exportador upstream faz `copytree` da Base inteira em **toda época**, depois
substitui pesos/config. Não usá-lo assim com teto de40GB e orçamento atual perto
de36GB. Compartilhar somente assets imutáveis do codec e tokenizer, escrever
configs novas e pesos temporários por rename. Nunca reabrir para escrita um
hardlink de config/peso de produção. Um novo export acústico BF16 demanda cerca
de1,812GB; checkpoints completos acrescentam estados do otimizador.

## Memória e rota de piloto

Contagem pelo cabeçalho safetensors, sem alocar o modelo: **914.643.008**
elementos, sendo8.854.336 do speaker encoder,311.164.928 do embedding de texto,
141.570.304 do preditor residual e453.053.440 dos demais tensores talker.
O apelido0.6B não significa que todo o arquivo contenha somente600M parâmetros.
O gerador excluindo speaker encoder tem aproximadamente905,789M elementos.

| Configuração analítica | Pesos BF16 + gradientes + momentos, sem ativações |
| --- | --- |
| Gerador completo, Adam com dois momentos BF16 | cerca de7,25GB |
| Gerador completo, dois momentos FP32 | cerca de10,87GB |
| Embedding de texto congelado, momentos FP32 | cerca de7,78GB incluindo todos os pesos BF16 |

O codec FP32 acrescenta aproximadamente0,682GB se mantido na GPU (outra precisão
exige verificação própria). Buffer do codec, contexto CUDA, logits, ativações e workspace precisam entrar
na medição. AdamW PyTorch2.6 inicializa momentos com `zeros_like(parameter)`:
no modelo BF16 direto, eles também podem ser BF16, sem master FP32 automático.
Não confundir essa economia com garantia de estabilidade em LR pequeno.
Congelar o embedding de texto é uma ablação local plausível que preserva o
léxico aprendido e deixa aproximadamente594,624M elementos ajustáveis; não
é a receita oficial sem modificações.

Ordem futura: microbatch1 totalmente GPU, acumulação, SDPA/checkpoint de
ativações e `use_cache=False` no treino; demonstrar update real, custo por passo,
checkpoint, processo novo/retomada e síntese inédita. Se não couber, estados
FP32 em CPU são válidos: aproximadamente7,25GB só nos dois momentos completos,
mais master/scratch se implementados. Operar por buckets para não duplicar
gradientes inteiros em RAM. Os23GiB do host tornam essa linha plausível, sem
provar encaixe ou custo. A inferência posterior deve manter gerador e decoder
integralmente na A2000, sem offload de pesos.

O script upstream não tem validação/early stop, retomada de otimizador/RNG,
scheduler nem salvamento atômico; seus exports não são checkpoints de retomada.
Um último estado BF16+Adam BF16 fica aproximadamente em5,45GB; último+
temporário+melhor export acústico exigem cerca de12,7GB adicionais no pior caso.
Momentos/master FP32 podem exigir muito mais. **Não cabe no orçamento de trabalho
atual sem limpeza planejada de experimentos encerrados.** Essa é uma condição
de preparação, não evidência de impossibilidade da arquitetura.

## TTFA sem duplicar DSP

No [wrapper de inferência](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_model.py),
`generate_custom_voice` devolve o áudio depois da geração completa; a própria
documentação esclarece que `non_streaming_mode=False` simula entrada de texto
incremental e não entrega geração em streaming. Não atribuir TTFA incremental
a essa chamada. Para textos longos, remover WORLD e ainda aguardar o WAV pode
ser pior que a produção atual.

A rota operacional a investigar depois do gate é carregar o export CustomVoice
em runtime isolado que realmente entregue códigos/PCM causal (verificar suporte
do vLLM-Omni atual ou implementar adaptador testado), com perfil novo como
`AegisVoice-v2-QwenSFT-experimental`. Não alterar configuração/voz protegidas.
Benchmarkar raw e eventual DSP com custo integral, mesmos textos e regra de
silêncio/TTFA. A hipótese só vence com ganho material de400ms definido no
protocolo e sem regressão de clareza/identidade. Não há medição nova de TTFA,
treino ou qualidade Qwen desta pesquisa.

Fontes pequenas e hashes preservados em
`/var/tmp/aegis-v2-work/qwen-sft-research/UPSTREAM.json`;
contagem em `header-parameter-counts.json`; inventário local completo em
`/var/tmp/aegis-v2-work/operations/qwen-model-inventory.json`.
