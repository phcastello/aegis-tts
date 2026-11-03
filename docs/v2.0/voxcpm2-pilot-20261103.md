# VoxCPM2: terceira linha preparada, 03/11/2026

**Hipótese:** o backbone multilíngue com LoRA no LM e no DiT pode aprender a
identidade Aegis mantendo o idioma e a emissão incremental; avançar somente se
AudioVAE preservar os atributos robóticos, o modelo completo sintetizar na GPU
sem offload de pesos e o piloto real atualizar/retomar/sintetizar em 12 GB.
Essa preparação não é um resultado de treino nem uma medição de VRAM/latência.

Código limpo clonado em `/var/tmp/aegis-v2-work/voxcpm2/source`, commit
`cce58a5b59303c9bd63d12f23afe6a49a4a80c59`, cerca de 11 MB. Pesos fixados em
`openbmb/VoxCPM2@32279effe8c19989596f05d353d1447f51d9e915`. Card, configuração e tokenizer ocupam cerca de 3,7 MB. Os dois pesos
necessários foram depois obtidos e validados por SHA-256 LFS remoto, SHA
incremental e releitura local antes de rename atômico. Metadados ficam em
`metadata/`, com links em `model/`; pesos finais ficam em `model/`. Não foi
criado ambiente nem cache duplicado. `download-receipt.json` registra a cópia
local íntegra; isto não é uma cópia remota recuperável de pesos treinados.

## Licença, idioma e representação acústica

Código Apache-2.0 pelo [LICENSE fixado](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/LICENSE).
O [card fixado dos pesos](https://huggingface.co/openbmb/VoxCPM2/blob/32279effe8c19989596f05d353d1447f51d9e915/README.md)
declara Apache-2.0 e português (`pt`) entre os idiomas. A API oficial do Hub e
o card baixados confirmaram essa declaração; não há arquivo `LICENSE` separado
nesse snapshot dos pesos (HTTP 404 registrado na inspeção). O AudioVAE faz parte
da mesma distribuição; denoiser opcional é excluído desta receita. Não há
promessa específica de sotaque brasileiro: validar PT-BR nas frases novas antes
de investir na especialização.

O [modelo nativo](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/model/voxcpm2.py)
recebe referência a 16 kHz e produz 48 kHz. O VAE fica congelado no SFT/LoRA
oficial. Portanto efeitos específicos removidos acima de 8 kHz não são
supervisionados diretamente pelo treino acústico; super-resolução aprendida não
prova recuperação da banda específica Aegis. O roundtrip é a primeira decisão,
com alvo reservado, controle encoder16k e VAE48k, sem ganho, denoise ou trim.

## Dependências e CUDA

`training/v2_voxcpm_probe.py --action imports` **passou em CPU**, com o ambiente
F5 existente: Torch/torchaudio 2.6.0+cu118, transformers4.49.0, einops0.8.2,
librosa0.11.0, pydantic2.14.2, safetensors0.8.0 e soundfile0.14.0. Nenhum pacote
foi instalado e nenhum ambiente/camada/imagem foi criado. A importação do
modelo e VAE nativos funciona sem Gradio, ModelScope, FunASR, WeText, denoiser
ou TorchCodec. O packer imutável é importado diretamente pelo arquivo: a
inicialização do pacote `voxcpm.training` importa o dataloader/argbind que não
é usado pelo piloto. Não se substituíram módulos com stubs.

O upstream publica PyTorch≥2.5 e CUDA≥12; o ambiente reutilizado compila com
CUDA11.8. A importação CPU não comprova que todos os kernels necessários
executem na A2000. O primeiro smoke CUDA deve registrar essa diferença e
falhar claramente se houver incompatibilidade. O caminho inicial é eager,
sem Triton/JIT/torch.compile. Não atualizar Torch ou driver de produção como
efeito colateral. O tokenizer local foi verificado em CPU separadamente: os sete casos de
validação têm 14–145 tokens, zero `<unk>`. O Transformers avisa que o nome de
classe do snapshot é VoxCPM2Tokenizer enquanto o loader usa LlamaTokenizerFast;
essa revisão o próprio upstream usa explicitamente LlamaTokenizerFast. O
wrapper manteve esse caminho e os arquivos fixados, sem suprimir o aviso. Não
é uma avaliação de pronúncia.
[Dependências oficiais](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/pyproject.toml).

## Verificação de carga CPU concluída

`--action cpu-check` passou com os pesos reais, CUDA oculto e sem consultar a
GPU: **42,162 s**, pico RSS **10.975.297.536 bytes**. O preflight mediu RAM
antes da carga e exigiu 11 GB de pico estimado mais 2 GB de reserva. Houve carga
completa BF16 e validação exata de **889 chaves base/VAE**, formas e amostra de
valor carregado; **384 tensores LoRA / 4.521.984 parâmetros FP32**. Totais de
parâmetros: 2.290.004.544 elementos BF16 e 98.735.938 FP32 (VAE+LoRA).
Uma LoRALinear real sob autocast CPU manteve saída BF16 e produziu gradiente
FP32 finito, sem alterar pesos. Isso verifica tipo/carga e uma operação; não é
forward completo, update, síntese ou viabilidade CUDA do modelo.

O AudioVAE também carregou separadamente em CPU com `strict=True`, zero chaves
faltantes/inesperadas, 94.213.954 parâmetros, 1,558 s e pico RSS 1,569 GB.
`torch.cuda.is_initialized()` ficou falso em ambos os recibos. Os avisos
públicos de tokenizer e weight_norm foram preservados nos logs.

Recibos: `voxcpm2/cpu-check/cpu-check.json`, `cpu-check-preflight.json`,
`cpu-check.log` e `vae-cpu-load.json`. O fonte exato validado foi preservado em
`cpu-check/validated-source.py`, SHA-256
`f141f57173c51a3e0d39748a161b6e6a546915f212761fb921b9f1a42cf275ae`.
A retomada/inferência exige conjunto LoRA completo, shapes/dtypes exatos e
valores finitos, antes de copiar. A ida à GPU verifica parâmetros, buffers,
caches KV e atributos Tensor comuns; derivados de weight_norm são recomputados
no destino antes da inspeção. Esses checks GPU ainda não foram executados.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_voxcpm_probe --action cpu-check \
  --run /var/tmp/aegis-v2-work/voxcpm2/cpu-check
```

## Memória e armazenamento: estimativas, não medições

| Artefato | Bytes / SHA-256 |
| --- | --- |
| model.safetensors | 4.580.080.592 / `f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d` |
| audiovae.pth | 376.951.122 / `94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1` |

Total de pesos: **4.957.031.714 bytes**. Estimar cerca de **5,4 GB adicionais**
para pesos, metadados, roundtrips, logs limitados e os checkpoints do piloto.
Baixar diretamente para `.part` no volume final e conferir hash antes de rename;
não manter simultaneamente cópia Hub, cache de download e cópia final. Nenhum
checkpoint do piloto deve duplicar o modelo base completo. Recalcular o total
único do projeto, incluindo pico simultâneo de checkpoints FastPitch/VITS e
temporários, antes do download. Se a soma superar 40 GB, retirar apenas artefatos
superados já identificados pelo supervisor; preservar best/último retomável,
fontes privadas e rollback. O agente de preparação não apagou artefatos. A limpeza de F5 superado foi
conduzida separadamente pelo supervisor, preservando best e estado retomável.

Gate de download medido em 06:52 UTC: 30.311.276.544 bytes únicos de projeto,
4.957.031.714 bytes por baixar e reserva de sobreposição de 2.500.000.000 bytes
→ pico de 37.768.308.258 <40.000.000.000 bytes. Volume tinha 33.244.168.192
bytes livres, mínimo exigido 5 GB. Ao terminar, o projeto usava
**35.268.435.968 bytes**, com **28.286.521.344 bytes livres** no volume.
`training/v2_voxcpm_fetch.py` mede inodes/blocos únicos nos mesmos roots do
inventário global, verifica metadados LFS e faz download direto em `.part`,
fsync, releitura, rename e fsync do diretório. Recibos em `download-plan.json`,
`download-progress.json` e `download-receipt.json`. Os picos de treino futuros
precisam continuar cabendo na reserva; não interpretar esse snapshot como
permissão para outros downloads fora do teto.

A configuração tem LM28 + residual8 camadas e DiT12. LoRA rank8 em Q/K/V/O de
LM+DiT implica **4.521.984 parâmetros** segundo as formas reais dos 192 lineares
no header íntegro; a instância deve confirmar isso. O arquivo base contém
2.290.004.544 elementos BF16, 4.580.009.088 bytes de payload. LoRA FP32 + gradientes + dois momentos Adam ocupa cerca de
72 MB; estados Adam/master na CPU custariam cerca de 54 MB de RAM e economizariam
pouco VRAM nesse rank. Preferir optimizer GPU primeiro, depois CPU se necessário.
O cache estático dos dois LMs soma aproximadamente 288 MiB em BF16/8.192 posições.
Ativações, VAE e workspaces continuam exigindo medição de pico.

Há um detalhe decisivo no loader oficial: `training=True` mantém o backbone
em FP32. O piloto preparado constrói/carrega o backbone congelado BF16 na CPU,
mantém VAE e LoRA FP32 e então move **todos** os pesos para CUDA. Isso reduz o
custo fixo estimado do backbone, sem quantização. Os caches KV são objetos Python,
não buffers registrados: o wrapper os recria explicitamente em CUDA depois de
`.to()`. Inferência verifica todos os parâmetros e caches na GPU. A adaptação
usa autocast BF16, microbatch1, acumulação opcional e checkpoint não reentrante
em cada bloco Transformer; não depende de inputs congelados terem gradiente.

A [estimativa oficial de treino](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html)
para LoRA v2 é maior que 12 GB em outra configuração. O nosso limite por sequência,
BF16 e microbatch menor justificam testar; **não comprovam encaixe**. Começar com
falas de treino completas de 3–6 s e até 512 tokens, sem recortar o áudio sob uma
transcrição inteira. Não usar essa seleção curta como receita final de voz.

## Ordem e comandos preparados

Todos os comandos CUDA abaixo são **preparados, não executados**. Precisam entrar
na fila GPU exclusiva já autorizada pelo mandato; o wrapper não para serviços,
não baixa pesos e recusa outro processo CUDA. Antes, confirmar hashes locais e
margem de disco/VRAM/RAM. Usar o Python existente:

```bash
python3 training/v2_voxcpm_probe.py --action plan
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  training/v2_voxcpm_probe.py --action imports
```

1. **AudioVAE sozinho**, usando os pesos locais já íntegros. Preservar as
   mesmas três âncoras de validação usadas no VITS posterior (0645/0040/0910),
   sem tocar o teste final. O supervisor já preparou o manifesto imutável
   `anchors.validation.jsonl` e o template `NEXT-voxcpm2-roundtrip.json`, sem
   enfileirar enquanto VITS ocupa a GPU:

```bash
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python -m training.v2_voxcpm_probe \
  --action roundtrip --gpu-authorized \
  --manifest /var/tmp/aegis-v2-work/voxcpm2/anchors.validation.jsonl \
  --run /var/tmp/aegis-v2-work/voxcpm2/roundtrip
```

2. **Inferência base completa na GPU**, com os dois arquivos locais já
   íntegros. O comando inclui saída nativa FLOAT48k, conversão24k, PCM/WAV e
   regra digital comum. Repetir a coorte de validação em um runner dedicado
   antes de aprovar o idioma; uma única frase só é smoke de carregamento.

```bash
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python -m training.v2_voxcpm_probe \
  --action infer --gpu-authorized --run /var/tmp/aegis-v2-work/voxcpm2/base-smoke \
  --text 'A válvula azul permanece fechada durante a revisão.'
```

3. **Piloto de pesos reais e retomada em processo novo**: LoRA LM+DiT, rank8,
   batch1, acumulação1, optimizerGPU primeiro. Salva a cada update; exige hash
   alterado de parâmetros treináveis, loss/gradiente finitos e síntese ao final.

```bash
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python -m training.v2_voxcpm_probe \
  --action pilot --gpu-authorized --steps 3 --rank 8 --optimizer-device cuda \
  --run /var/tmp/aegis-v2-work/voxcpm2/pilot
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python -m training.v2_voxcpm_probe \
  --action pilot --gpu-authorized --steps 4 --rank 8 --optimizer-device cuda \
  --resume /var/tmp/aegis-v2-work/voxcpm2/pilot/step-0003.pt \
  --run /var/tmp/aegis-v2-work/voxcpm2/pilot
```

Checkpoint próprio contém LoRA, AdamW, scheduler, precisão/scaler (BF16 sem
GradScaler), RNG Python/NumPy/Torch/CUDA, passo, cursor, seleção e hashes.
Escreve temporário no mesmo volume, fsync, recarrega e confere conteúdo antes
de rename. Cada checkpoint tem recibo SHA-256; a retomada valida integridade e
configuração antes de carregar. O código oficial mantém cópias `step/latest`
e não restaura todo esse estado; o wrapper não usa essa persistência incompleta.
A retenção do piloto é limitada por no máximo dez updates. Antes de treino
longo, implementar retenção last/best, validação fixa, paciência e supervisor;
não transformar o loop curto em treino de vários dias sem essas etapas.

Se optimizerGPU/ativação não couber, repetir uma **nova** tentativa com
`--optimizer-device cpu`, microbatch1 e acumulação. Não reaproveitar checkpoint
com configuração incompatível sem migração explícita. Se a inferência completa
não couber em CUDA ou a reconstrução destruir o caráter robótico sem solução
medida, encerrar a linha e limpar os pesos exclusivos depois de guardar evidência.
Se os gates passarem, medir o streaming nativo usando o adaptador existente
`benchmarks/architecture_compare/voxcpm_runtime.py`, adaptando o formato final
24k e seu custo antes da comparação de produto. Não usar números da 4090 para
concluir vantagem nesta A2000.

Validação realizada: checkout limpo/pin, pesos/metadados/hashes oficiais, tokenizer e
imports CPU, plano sem GPU e compilação Python. Não houve forward/backward completo do modelo, update do otimizador,
checkpoint treinado ou áudio VoxCPM2. O backward CPU restrito à LoRALinear real
é descrito acima; não o contar como piloto de treino da candidata.

Revisão independente CPU de engenharia passou: packer real com três comprimentos
sintéticos e VAE simulado, máscaras/labels e padding corretos; wrapper real de
checkpoint em dois blocos minúsculos com inputs congelados preservou gradientes
distintos (sem captura tardia da camada). É teste de contrato, não treino da voz.
O recibo público compacto está em `benchmarks/v2/voxcpm2-preflight-20261103.json`,
sem áudio, transcrição privada, estados RNG ou pesos. A única mudança após a
carga CPU validada foi exigir saída VAE finita, não vazia e com comprimento
padded exato antes de retirar padding conhecido, preservando FLOAT de falha.
Não foi repetida a carga CPU por essa mudança restrita ao futuro roundtrip.
