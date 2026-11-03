# Artefatos locais Qwen e rollback — 03/11/2026

Inventário somente leitura, CPU nice19, sem carregar modelo ou alterar serviço,
referências, caches ou configuração. Raiz real:
`/home/phc/aegis-tts/models/Qwen3-TTS-12Hz-0.6B-Base/`.

Os40arquivos somam2.516.107.412bytes; todos foram hashados integralmente,
sem pointers LFS. Os headers safetensors carregam e os offsets cabem nos
arquivos. A revisão Hugging Face local
`5d83992436eae1d760afd27aff78a71d676296fc` coincide nos metadados de download
dos pesos e em `cache/huggingface/hub/.../refs/main`.

| Artefato | Bytes | Armazenamento | SHA-256 |
| --- | ---: | --- | --- |
| `model.safetensors` |1.829.344.272 |478tensores BF16 |`180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6` |
| `speech_tokenizer/model.safetensors` |682.293.092 |496tensores FP32 |`836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258` |

Não há config de quantização inteira nem tensores qweight/qzeros/scales nesses
pesos. O dtype de armazenamento não determina sozinho todos os dtypes do
runtime. A arquitetura local é `Qwen3TTSForConditionalGeneration`, talker
hidden1024/28layers/16heads/8KVheads, code predictor5layers,16grupos.
Tokenizer12Hz V2 recebe e produz24kHz. A versão4.57.3 nos configs é a versão
de exportação, não uma medição da versão atualmente instalada no container.

O cache vLLM-Omni contém3.204arquivos, cerca de308,4MB de blocos de arquivos
alocados; o inventário de caches lê apenas paths/tamanhos/refs públicos,
sem conteúdo de credenciais ou prompts. Os diretórios e arquivos auxiliares
de produção permanecem preservados. O modelo1.7B preexistente também foi
preservado e não foi confundido com o mount operacional0.6B.

Recibo privado completo, incluindo config selecionada, SHA de todos os
arquivos, realpaths/inodes, metadados HF e caches:
`/var/tmp/aegis-v2-work/operations/qwen-model-inventory.json`.
Reprodução: `python3 -m training.v2_qwen_inventory --output CAMINHO_PRIVADO`.
Serviço/imagem/headers e hashes das referências têm seus snapshots
operacionais próprios; o inventário não substitui o smoke de rollback.
