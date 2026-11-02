# Qwen 0.6B e VoxCPM2: evidência técnica para a comparação

Consulta em 02/11/2026, somente metadados e código público pequeno. Nenhum
peso baixado, modelo carregado, treino ou benchmark CUDA executado nesta pesquisa.
Valores publicados em outra GPU não são resultados da RTX A2000.

## Proveniência verificável

| Componente | Revisão consultada |
| --- | --- |
| OpenBMB/VoxCPM | `cce58a5b59303c9bd63d12f23afe6a49a4a80c59` |
| openbmb/VoxCPM2, Hugging Face | `32279effe8c19989596f05d353d1447f51d9e915` |
| QwenLM/Qwen3-TTS | `022e286b98fbec7e1e916cb940cdf532cd9f488e` |
| Qwen/Qwen3-TTS-12Hz-0.6B-Base, Hugging Face | `5d83992436eae1d760afd27aff78a71d676296fc` |
| Qwen/Qwen3-TTS-Tokenizer-12Hz, Hugging Face | `7dd38ad4e9bad454aae9cd937d0cd577604fe229` |

Os pins upstream não substituem os hashes do modelo local já implantado. A
produção permanece no runtime/imagem registrado em
[production-latency-20261101.md](../v0.8.0/production-latency-20261101.md).

## Qwen3-TTS-12Hz-0.6B-Base

O código oficial é Apache-2.0, os pesos Base também são declarados Apache-2.0
no card, e o tokenizer/decoder tem a mesma declaração em seu próprio card.
Não aplicar a licença do runtime aos pesos por associação. Fontes:
[LICENSE do código](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/LICENSE),
[card Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/5d83992436eae1d760afd27aff78a71d676296fc/README.md),
[card do tokenizer](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/blob/7dd38ad4e9bad454aae9cd937d0cd577604fe229/README.md).

A arquitetura combina geração autoregressiva de códigos acústicos, predição
dos demais codebooks e decoder neural causal. O codec utiliza 16 codebooks e
12,5 quadros/s; o nome comercial 12Hz não é frequência de áudio. A configuração
define **entrada e saída de 24.000 Hz**. A API Python devolve arrays float32;
PCM16 é a serialização do serviço atual, não uma resolução acústica intrínseca
de 16 bits. O limite de Nyquist é 12 kHz; salvar em 48 kHz não recria a banda
acima disso. Fontes:
[config do codec da Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/5d83992436eae1d760afd27aff78a71d676296fc/speech_tokenizer/config.json),
[API do tokenizer](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/inference/qwen3_tts_tokenizer.py).

O suporte oficial nomeia **Portuguese**, sem uma garantia separada de sotaque
brasileiro. A implantação Aegis é a evidência local de utilização PT-BR; a
aceitação de pronúncia e identidade nas frases da nova comparação exige escuta.
Streaming incremental existe no codec e no runtime operacional. O tempo
promocional de 97 ms não representa o primeiro PCM útil com WORLD, reserva,
fila e HTTP na A2000. Fontes:
[README oficial](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/README.md),
[medição local](../v0.8.0/production-latency-20261101.md).

O fluxo oficial de fine-tuning suporta um locutor: extrai `audio_codes` e
ajusta o gerador com `sft_12hz.py`. Ele **não treina o codec** nessa receita;
portanto não remove perdas de banda ou efeitos na reconstrução do tokenizer.
O script usa bf16, AdamW, batch padrão 2 e acumulação 4; não apresenta garantia
de memória para A2000. Inferência caber não comprova que SFT caiba. Viabilidade
de treino em 12 GB permanece **não medida**, e esta etapa não deve executar
passos de treino para descobri-la. Fontes:
[guia SFT](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/README.md),
[script SFT](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning/sft_12hz.py).

O experimento 1.7B acelerado já foi rejeitado pelo proprietário por inviabilidade
operacional de VRAM; não foi repetido nem é uma candidata desta comparação.

## VoxCPM2

Código, modelo e AudioVAE distribuído no mesmo repositório de pesos são
declarados Apache-2.0. A licença da versão inspecionada pode ser conferida no
[LICENSE do código](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/LICENSE)
e no [card dos pesos](https://huggingface.co/openbmb/VoxCPM2/blob/32279effe8c19989596f05d353d1447f51d9e915/README.md).
O denoiser opcional é outro modelo e fica desligado na receita de comparação;
sua licença e memória não são implicitamente as do VoxCPM2.

O backbone de 2B usa modelagem autoregressiva e difusão sobre latentes contínuos,
com LM base/residual e LocDiT. A configuração inclui LM bf16 e AudioVAE FP32.
O gerador nativo entrega arrays float32 mono em **48.000 Hz**. A
[configuração do checkpoint](https://huggingface.co/openbmb/VoxCPM2/blob/32279effe8c19989596f05d353d1447f51d9e915/config.json)
fixa encoder 16.000 Hz, decoder 48.000 Hz e cache de contexto máximo 8.192;
o dtype dos pesos não é profundidade de áudio PCM.

**48 kHz de saída não significa preservar os efeitos da referência até 24 kHz.**
O caminho de clonagem carrega a referência com `librosa.load(sr=16000)`, antes
do encoder; conteúdo acima de 8 kHz não chega diretamente ao condicionamento.
O decoder faz super-resolução aprendida, capaz de gerar banda alta, mas sem
garantia de reproduzir a banda alta específica do efeito Aegis. Esse risco é
mais importante aqui que o número nominal de Hz. Fontes:
[carregamento da referência e dtype](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/model/voxcpm2.py),
[relatório técnico](https://arxiv.org/abs/2606.06928).

Português figura entre as 30 línguas oficiais; não há garantia separada de
PT-BR no card nem nos tags (`pt`). A validação de sotaque, siglas e números
brasileiros continua aberta. Além disso, o normalizador opcional do código
seleciona chinês ou **inglês**; `normalize=True` não é frontend PT-BR.
A comparação deve usar `normalize=False` e registrar à parte a normalização
Aegis, com trilhas de texto literal e fala expandida. Fonte:
[normalizador](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/utils/text_normalize.py).

Streaming incremental existe no código: o LM emite latentes por passo e
`streaming_decode()` mantém estado causal entre chunks. Não é apenas cortar
um WAV completo em pacotes. A API oferece `generate_streaming`; o custo da
preparação da referência deve entrar no tempo até PCM, salvo uma medição
explicitamente separada com referência em cache. `retry_badcase` não funciona
em streaming, e deve ficar falso nos dois modos para comparação justa. Fontes:
[decoder incremental](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/modules/audiovae/audio_vae_v2.py),
[API pública e saída numpy](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/core.py).

O README publica aproximadamente **8 GB para inferência nativa** e RTF ~0,30
em RTX 4090, sem uma medição A2000 equivalente. São pistas de que merece um
teste isolado, não confirmação de encaixe ou baixa latência. A receita oficial
vLLM-Omni documenta 4090/24 GB, com aproximadamente 22 GB residentes no padrão;
ela não cabe nesta A2000 sem outra configuração. Não se escolheu vLLM como
pré-requisito. Fontes:
[tabela upstream](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/README.md#-model-details),
[receita vLLM-Omni](https://github.com/vllm-project/vllm-omni/blob/main/recipes/OpenBMB/VoxCPM2.md).

A primeira configuração a aferir é o PyTorch nativo: pesos locais, dispositivo
explícito `cuda:0`, `local_files_only=True`, `load_denoiser=False`, sem
normalização nem denoise. Começar com `optimize=False`. A revisão inspecionada
compara o device literalmente com `cuda` no caminho de compilação e pode pular
a otimização com `cuda:0`, emitindo só um warning; o adaptador recusa
`--optimize` para evitar registrar aceleração inexistente. Um futuro ensaio
compilado exige revisão própria e confirmação do backend realmente ativo,
incluindo carga/warm-up e picos adicionais. Fontes do detalhe:
[optimize](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/model/voxcpm2.py#L255),
[device resolver](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/model/utils.py#L192).
As dependências publicadas exigem Torch >=2.5 e CUDA >=12;
**não** instalar simplesmente o Torch mais recente: driver 535 e kernels
Triton/PTX exigem compatibilidade efetiva com sm86. Compatibilidade menor de
CUDA12 com driver >=525 tem limitações para PTX/JIT. Fontes:
[dependências](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/pyproject.toml),
[FAQ de runtime](https://voxcpm.readthedocs.io/en/latest/faq.html),
[compatibilidade NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Os dois arquivos grandes somam 4.957.031.714 bytes (~4,62 GiB):
`model.safetensors` 4.580.080.592 e `audiovae.pth` 376.951.122, além do tokenizer
e configuração. Não incluem ambientes, rodas CUDA, caches compilados e
espaço temporário de instalação. A máquina tinha apenas ~17 GB livres na
inspeção inicial: dimensionar tudo antes de qualquer download. Tamanhos e
hashes foram consultados na
[árvore da revisão HF](https://huggingface.co/api/models/openbmb/VoxCPM2/tree/32279effe8c19989596f05d353d1447f51d9e915),
sem baixar os arquivos. SHA256 LFS: modelo
`f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d`;
VAE `94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1`.

## Fine-tuning VoxCPM2: separado da inferência

SFT e LoRA existem para v2, com configurações próprias. A tabela `Hardware`
do guia identifica explicitamente a linha **VoxCPM 2 (2B)** e estima
**~20 GB LoRA / ~40 GB full** para batch 16 e teto 8.192 tokens; não são medições
na A2000. Batch menor e amostras curtas podem reduzir memória, mas não autorizam
declarar 12 GB viável. O exemplo YAML corrente usa batch 2 e acumulação 8;
isso não transforma a estimativa de treino em medição dessa configuração.
Fontes:
[guia de treino](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html),
[config LoRA v2](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/conf/voxcpm_v2/voxcpm_finetune_lora.yaml).

O código do treino reamostra entradas ao encoder de **16 kHz** e congela o
AudioVAE, inclusive no modo full. Ajustar LM/DiT não ensina esse decoder a
recuperar efeitos específicos eliminados pelo condicionamento. Antes de
qualquer decisão de treino, comparar alvo original, versão limitada ao encoder
e reconstrução VAE, com escuta de identidade/aspereza/ataques/sibilantes.
Treinar outro VAE é uma proposta arquitetural distinta, sem receita validada
na A2000. Fontes:
[trainer](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/scripts/train_voxcpm_finetune.py),
[congelamento em from_local](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/model/voxcpm2.py#L1169).

Conclusão desta pesquisa: VoxCPM2 é **exploratório**, sujeito primeiro a memória,
runtime e reconstrução na janela autorizada. Não há resultado local que prove
melhora sobre Qwen, e o gargalo de 16 kHz enfraquece sua prioridade para efeitos
que o proprietário exige preservar. A seleção final depende de medições e A/B,
não do rótulo 48 kHz.

## Runtime isolado preparado, ainda sem validação CUDA

[`voxcpm_runtime.py`](../../benchmarks/architecture_compare/voxcpm_runtime.py)
implementa a API nativa verificada acima, exposta somente em loopback, porta
isolada 18083, com HTTP streaming PCM16. O modo padrão `--plan` não importa
Torch/VoxCPM nem toca a GPU. `--execute` exige pesos e metadados locais fixados,
checkout upstream limpo no pin, referência/transcrição privadas, diretório de
saída 0700 e janela aprovada específica `voxcpm2-native`.

Antes de importar o runtime, ele recusa qualquer processo CUDA e VRAM ocupada;
não interrompe produção. Um watchdog encerra somente esse processo quando a
janela expira, inclusive durante carga/compilação. O cliente de benchmark
confere exclusividade durante as requisições. O servidor desliga a normalização
linguística opcional; a API nativa ainda uniformiza espaços e quebras de linha.
Não aplica ganho, corte de silêncio ou reamostragem de saída. Valores float fora
de [-1,1] causam falha explícita. `--save-float` preserva arrays float32 privados
para análise. A terminação HTTP chunked só é enviada após o gerador terminar;
uma falha anterior interrompe o transporte e invalida aquela medição. Esse fim
não prova EOS semântico: o limite `max_len` pode encerrar uma fala incompleta.
O receipt registra essa limitação e a cauda precisa de escuta.

O condicionamento preparado é `continuation` (referência + transcrição),
explicitamente gravado no receipt. O cliente inclui os checks de GPU/janela
no tempo HTTP; o receipt nativo separa o tempo desse guard da geração.
Artefatos de diagnóstico são gravados depois do EOF, com tempo separado.
Medições de desempenho deixam `--save-float` desligado, pois o I/O anterior
pode atrasar a aceitação do próximo pedido. Usar uma rodada de qualidade
separada para conservar float32. Os loaders recusam pesos alternativos que
o upstream poderia priorizar sem conferir os hashes declarados.

Validação feita nesta etapa: importação/compilação Python, `--plan` sem acesso à
GPU, recusa de snapshots ocupados, conversão PCM16 e HTTP incremental com gerador
simulado. Uma falha simulada após o primeiro chunk resultou em HTTP incompleto,
detectado pelo cliente, e os arrays float32 simulados foram salvos privadamente.
**Não** há inferência real nem garantia de que dependências/compilação caibam
na máquina; registrar isso antes de marcar a configuração `runtime_verified`.

```bash
# Inspeção sem CUDA, sem importar voxcpm e sem baixar nada:
python3 benchmarks/architecture_compare/voxcpm_runtime.py --plan

# Somente depois da instalação isolada dimensionada e da janela aprovada:
/AMBIENTE_ISOLADO/bin/python benchmarks/architecture_compare/voxcpm_runtime.py \
  --execute --window-file /PRIVADO/janela-voxcpm2.json \
  --source-dir /FONTES/VoxCPM --model-dir /PESOS/VoxCPM2 \
  --reference-wav /PRIVADO/referencia.wav \
  --reference-text-file /PRIVADO/referencia.txt \
  --output-dir /PRIVADO/voxcpm2-runtime --save-float
```

O modo padrão é eager (`optimize=False`); `--optimize` fica bloqueado até existir
um adaptador com backend compilado verificado. O exemplo acima é uma rodada
de qualidade, pois solicita `--save-float`; omitir a flag na rodada de latência.
Guardar margem na janela para a retomada operacional manual do serviço atual.

## Reconstrução AudioVAE: protocolo anterior a qualquer treino

O caminho abaixo corresponde à API consultada do
[AudioVAE V2](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/modules/audiovae/audio_vae_v2.py).
É um **snippet de protocolo, não executado**, para adaptar dentro da mesma
proteção de janela do runtime; não deve ser rodado como um comando CUDA avulso.
Usar clipe privado tratado, com autorização de uso, mono e curto. Preservar o
arquivo original e sua taxa/bit depth; não substituir o master pelo controle.

```python
# model já carregado SOMENTE na janela autorizada.
# source_f32 é o áudio mono original, e source_rate a taxa verdadeira do arquivo.
encoder_input = librosa.resample(
    source_f32, orig_sr=source_rate, target_sr=16000
).astype(np.float32)
vae = model.tts_model.audio_vae  # FP32; encoder espera 16 kHz
with torch.inference_mode():
    tensor = torch.from_numpy(encoder_input).reshape(1, 1, -1)
    tensor = tensor.to(model.tts_model.device, dtype=torch.float32)
    latent = vae.encode(tensor, 16000)
    reconstructed = vae.decode(latent).squeeze().cpu().numpy()
# encode faz padding ao hop. Retirar SOMENTE a cauda adicionada por esse padding:
reconstructed = reconstructed[:len(encoder_input) * 3]
encoder_control_48k = librosa.resample(
    encoder_input, orig_sr=16000, target_sr=48000
).astype(np.float32)
```

Escutar em ordem cega: original; controle limitado ao encoder; reconstrução
VAE48k; clonagem em frase inédita. O controle é explicitamente reamostrado,
não síntese de banda alta. Salvar float32 sem ganho automático e registrar
duração/padding, energia por banda, aspereza, ataques e sibilantes. Se o
controle 16k já eliminar o efeito exigido, o teste revela um obstáculo estrutural
antes de investir em SFT/LoRA; o modelo ainda pode gerar agudos novos, mas não
conserva os agudos específicos removidos da referência.
