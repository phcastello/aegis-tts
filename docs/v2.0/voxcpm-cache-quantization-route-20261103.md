# VoxCPM2: cache estático e quantização, 03/11/2026

A próxima ablação deve reduzir o cache estático de 8.192 para 1.024 posições,
mantendo pesos e cálculos de atenção. É uma hipótese derivada do código,
ainda sem perfil ou medição local desta variante. O controle nativo com cache
8.192 já medido pelo processo principal teve RTF longo 1,98994 e lacunas;
reduzir NFE de 10 para 4 pouco alterou a velocidade dos controles por arquivo.

No pin VoxCPM2 `cce58a5b59303c9bd63d12f23afe6a49a4a80c59`, o
`MiniCPMAttention.forward_step` atende o cache inteiro com uma máscara booleana
por posição e GQA de 16 cabeças de consulta para duas de K/V. São 28 camadas no
LM base e oito no residual. PyTorch 2.6 rejeita Flash SDPA com máscara não nula
e sua rota efficient não suporta essa diferença de cabeças. Isso justifica
investigar o custo do cache, mas **não identifica sozinho o kernel local**:
cuDNN/compilação e o perfil real ainda precisam ser observados.
[Guarda Flash](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/transformers/sdp_utils_cpp.h),
[seleção dos backends](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp).

`training/v2_voxcpm_cache_probe.py` é um novo wrapper. Altera apenas
`model.config.max_length` antes de o loader existente recriar ambos os caches
CUDA e antes de `optimize()`/tracing. A posição máxima do LM/RoPE, máscara,
GQA, implementação SDPA, precisão, pesos, NFE/CFG e conversão PCM seguem o
caminho original. O recibo inclui capacidades, shapes, dtype, dispositivo e
bytes reais dos dois caches. O KV nominal passa de 301.989.888 para
37.748.736 bytes; isso não prevê o ganho de velocidade.

A guarda está no `_generate` central, cobrindo arquivo e streaming nativo,
dentro do relógio de síntese. Exige
`tokens_texto + max_len_solicitado + 2 <= capacidade`; um slot é o início de
áudio e outro é margem. Prompts/referências são recusados nessa ablação.
Entradas grandes falham explicitamente; não há truncamento. Esse limite de
entrada é parte da nova configuração e precisa ser validado para uso real.

Cinco testes CPU passaram, incluindo as duas APIs, limites, restauração dos
hooks e equivalência float64 de GQA mascarado com prefixos iguais e futuros
arbitrários, em capacidades 16/128. CUDA não foi inicializado. Isso demonstra
a equivalência matemática do prefixo, sem prometer identidade numérica em GPU
ou igualdade das sínteses autoregressivas. Os scripts compartilhados e a fonte
upstream não foram editados. Plano privado:
`/var/tmp/aegis-v2-work/voxcpm2/cache1024-preparation/cache-plan.json`.

Comando futuro, somente pela fila GPU exclusiva, com CPU ASR suspenso:

```sh
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_voxcpm_cache_probe --action execute --gpu-authorized \
  --delivery native --kv-cache-capacity 1024 --nfe 4 --optimize \
  --cases v_reply v_long --repetitions 3 \
  --run /var/tmp/aegis-v2-work/voxcpm2/cache1024-native-nfe04
```

Não foi enfileirado nem executado em GPU por esta subtask. A comparação deve
usar os controles nativos 8.192, aquecimento por caso, TTFA, EOF, RTF longo,
lacunas e ASR das novas saídas. O teste final continua fechado.

## Quantização: manter como hipótese posterior

O caminho INT4 padrão do TorchAO 0.9 usa o tinygemm
`aten._weight_int4pack_mm`, com ativações BF16. No PyTorch 2.6 o kernel CUDA e
o empacotador são condicionados a `CUDA_VERSION >= 12000`; o build atual
`2.6.0+cu118` não os disponibiliza. A presença dos headers/nomes de operador
não prova execução. Instalar TorchAO não corrige essa restrição do build;
por isso não houve instalação. TorchAO tem licença BSD-3.
[Implementação TorchAO](https://github.com/pytorch/ao/blob/v0.9.0/torchao/dtypes/uintx/tensor_core_tiled_layout.py),
[guarda do kernel PyTorch](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/int4mm.cu),
[licença](https://github.com/pytorch/ao/blob/v0.9.0/LICENSE).

Uma alternativa compatível a investigar depois é bitsandbytes 0.50.2, MIT:
suporta Python 3.10+, Torch 2.4+, CUDA 11.8 e SM86. O wheel Linux x86-64 é
43.139.553 bytes, SHA256
`55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283`
segundo metadados PyPI; **não foi baixado nem validado localmente**.
Reserve 256 MiB para download/expansão, além do cache de compilação existente,
e use overlay `--no-deps --no-cache-dir`, sem duplicar os pesos de 5 GB.
[Requisitos oficiais](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation),
[metadados do release](https://pypi.org/project/bitsandbytes/0.50.2/#files),
[licença](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/LICENSE).

Há 252 matrizes lineares dos dois LMs, com 1.698.693.120 elementos; os 150,4 M
elementos de embedding ficam fora dessa contagem. Apenas metadados safetensors
foram lidos. INT8/NF4 diminuiriam armazenamento/tráfego desses pesos, mas não
resolvem o custo de atenção sobre posições futuras. A implementação
`Linear4bit` ainda pula testes fullgraph em Torch anterior a 2.8; perder a
compilação atual pode anular o ganho. O novo operador fused 4bit do release
0.50 é uma possibilidade para um piloto específico, sem garantia de captura
CUDA Graph ou ganho nesta A2000.
[Teste primário de compatibilidade](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/tests/test_linear4bit.py),
[release e kernels](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.50.0).

Se chegar a esse piloto, testar primeiro INT8 e/ou NF4 somente nos dois LMs,
mantendo VAE/DiT na GPU, com kernels pequenos dos shapes reais, compilação e
replay comprovados, depois as mesmas frases. Não basta medir VRAM. Treinar
LoRA BF16 e depois fundir/quantizar para inferência cria uma nova variante que
exige ASR e identidade; QLoRA seria outro treino com novo piloto de update,
checkpoint e retomada. O treino BF16 já cabe, portanto QLoRA não é necessário
para a pergunta atual de latência.
