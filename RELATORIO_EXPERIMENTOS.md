# Relatório de experimentos de desempenho — Qwen3-TTS

Data: 2026-07-16  
GPU: NVIDIA RTX A2000 12 GB  
Modelo: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, BF16, SDPA

## Resumo

O diagnóstico confirma que a síntese é limitada pelo caminho autoregressivo
eager e pelo grande número de lançamentos de kernel. No teste representativo,
o `StaticCache` foi compatível com o Qwen e preservou o áudio, mas piorou a
mediana de RTF em 1,2%. Não houve evidência de compilação automática,
recompilação ou CUDA Graphs.

| Experimento | Resultado principal | Decisão |
| --- | --- | --- |
| 1. PyTorch Profiler | ~46 mil launches para uma frase curta; `aten::mm` domina CUDA | Investigar redução de launches/compilação regional. |
| 2. Baseline dinâmico | Mediana de RTF 5,2434 | Referência representativa para testes seguintes. |
| 3. Static KV Cache | Mediana de RTF 5,3072; 1,2% pior; áudio idêntico | Não promover para produção. |

## 1. Perfil PyTorch do caminho normal

Configuração: uma síntese controlada com perfil CPU+CUDA, `profile_memory=True`,
voz `giseli-10s` e texto curto. A trace está em
[`samples/qwen-profile.json`](samples/qwen-profile.json) (137 MB).

| Métrica | Medição |
| --- | ---: |
| Intervalo coberto pela trace | 5,868 s |
| Self CPU total | 3,935 s |
| Self CUDA total | 441,867 ms |
| `cudaLaunchKernel` | 1,058 s de CPU; 46.409 chamadas |
| `aten::mm` | 199,208 ms CUDA; 45,08% do CUDA |
| `_flash_attention_forward` | 6,032 ms CUDA; 1,37% do CUDA |

O `topk` apareceu 97 vezes; como há 46.409 launches, a ordem de grandeza é
~478 launches por passo de decoding. Assim, o problema não é atenção: SDPA já
escolheu o kernel Flash interno e ele responde por parcela pequena do tempo
CUDA. A prioridade é reduzir dispatch/fragmentação de operações no decoding.

O profiling teve overhead operacional alto: a exportação de 137 MB prolongou a
requisição, embora a trace cubra 5,868 s. Os valores do profiler devem ser
usados para distribuição de trabalho, não como latência de produção.

## 2. Baseline com cache dinâmico

Configuração: três sínteses sem profiler, voz `giseli-10s`, `seed: 41`, mesma
imagem/runtime e a frase representativa abaixo. Os valores são os headers
retornados pela API.

> Olá, Pedro. Pelo que eu verifiquei, todos os serviços estão funcionando
> normalmente. A reunião continua marcada para segunda-feira, e ainda há
> algumas mensagens que precisam de resposta. Depois disso, posso organizar as
> próximas tarefas e revisar o que ficou pendente para amanhã.

| Repetição | Geração | RTF |
| --- | ---: | ---: |
| 1 | 88,9717 s | 5,3469 |
| 2 | 87,2500 s | 5,2434 |
| 3 | 86,3320 s | 5,1882 |
| **Mediana** | **87,2500 s** | **5,2434** |

Cada execução produziu 16,6400 s de áudio. VRAM global antes do experimento:
~4,78 GB.

## 3. Static KV Cache

Configuração: imagem/contêiner isolados em `127.0.0.1:8002`,
`use_cache=True`, `cache_implementation="static"`, `TORCH_LOGS` com
`graph_breaks,recompiles,perf_hints`; mesma frase, voz e seed do baseline. A
primeira geração foi descartada: 88,1604 s (RTF 5,2981).

| Repetição | Geração | RTF |
| --- | ---: | ---: |
| 1 | 87,7908 s | 5,2759 |
| 2 | 88,3117 s | 5,3072 |
| 3 | 88,7719 s | 5,3348 |
| **Mediana** | **88,3117 s** | **5,3072** |

Comparação com o baseline:

| Métrica | Dinâmico | Estático | Variação |
| --- | ---: | ---: | ---: |
| Mediana de geração | 87,2500 s | 88,3117 s | +1,2% |
| Mediana de RTF | 5,2434 | 5,3072 | +1,2% |
| VRAM global após a série | ~4,78 GB | ~5,52 GB | +~0,75 GB |

Os seis WAVs (três dinâmicos e três estáticos) tiveram o mesmo SHA-256:
`a8d38468fc3c7dca3eafaa132dd0320f12403a476ea1bfd2bbd6377794bc4456`.
Logo, nesta amostra o cache estático não alterou o áudio gerado.

O Qwen aceitou o cache sem OOM, mas não houve logs de compilação, graph break
ou recompilação e o warm-up não foi excepcionalmente lento. A interpretação
mais provável é que o cache é usado, porém o hot loop customizado não entrou no
caminho de compilação automática do Transformers nesta versão. Como consome
mais VRAM e ficou marginalmente mais lento, o experimento está reprovado para
este runtime.

## Próximo experimento recomendado

Testar `torch.compile(..., mode="reduce-overhead")` somente após identificar o
`torch.nn.Module` do decoding autoregressivo. Repetir o perfil PyTorch e exigir
menos launches, kernels médios mais longos e RTF inferior a 5,6862 antes de
considerar a mudança válida.
