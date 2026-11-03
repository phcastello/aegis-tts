# VoxCPM2: preparação da inferência otimizada

O piloto já comprovou três updates, retomada em processo novo para o quarto e
síntese na A2000. Isso é engenharia de treino, sem conclusão sobre a qualidade
da voz. A medição eager anterior de12,226s para1,76s de áudio ocorreu fria, com
ASR CPU concorrente e192 módulos LoRA ativos mesmo com deltas zero. Não basta
para descartar a arquitetura por velocidade.

**A preparação inicial foi CPU; o supervisor executou depois os controles GPU
descritos abaixo.** A nova ablação NFE6/4 e TF32 deste registro ainda é somente
código/plano CPU. Não houve outra compilação GPU, download de Torch/ambiente ou
job enfileirado nesta extensão. FastPitch segue aprendendo; qualquer próxima
janela depende de checkpoint íntegro e retomada coordenada pelo supervisor.

## Resultado local do controle NFE10

Mesma A2000, Base sem LoRA, CFG2, seed40733133, precisão `highest`, caso
`v_reply`, três observações aquecidas, WAV completo24k e resampling incluídos:

| Configuração | EOF p50 | Áudio | RTF p50 | Estado |
| --- | --- | --- | --- | --- |
| Eager sem LoRA | 5,604s | 1,76s | 3,1840 | Não acompanha reprodução |
| Compilado oficial | 3,318s | 1,76s | 1,8855 | Acelera, mas ainda falha no gate de tempo real |

O compilado gerou613 kernels e12 nós CUDAGraph; o hook observou378 replays reais
nos warmups, sem erro de driver. O toolchain funciona nesta placa/configuração;
GPU fit e prova CPU não eram o gargalo final. Com somente três observações, p95
continua indisponível. Esses valores são do motor por arquivo, sem TTFA HTTP
ou streaming atribuído. Não há aprovação de pronúncia ou identidade.
Evidência compacta em `voxcpm2/{base-eager-controlled,base-compiled-controlled}/`
(`recipe.json`, `summary.json`, `results.jsonl` e WAVs pareados).

## Ablação mínima preparada:10,6,4 e TF32 separado

O argumento `--nfe` permite10/6/4. Ele é passado realmente a
`model.generate(inference_timesteps=...)` e registrado por chamada/receita.
O helper antigo usa `getattr(args,'nfe',10)`, preservando os pilotos sem flag.
CFG2, seed, limite de comprimento, ausência de retry/denoise e texto permanecem
idênticos. NFE6/4 são configurações novas, cuja qualidade não é presumida.

Aqui NFE é o rótulo do argumento upstream `inference_timesteps`. O CFGZeroStar
padrão pula o primeiro cálculo do estimador:10/6/4 correspondem a9/5/3 chamadas
DiT por patch. Isso foi confirmado com o CFM upstream real e um estimador
minúsculo CPU, sem inicializar CUDA; recibo
`voxcpm2/cfm-timesteps-cpu-proof.json`. O runner também contará as chamadas
reais do estimador, sem confundir passos solicitados com avaliações executadas.

Aquece **cada caso** com três calls excluídos antes de começar qualquer
observação da matriz. Novos shapes de `v_pressure`/`v_long` não serão rotulados
como aquecidos na primeira chamada. O replay hook cobre esses warmups e é
restaurado antes dos tempos warm; o contador acumulado não representa replays
medidos durante as repetições cronometradas.

Primeiro executar somente `v_reply`, n3, `highest`, em processos novos: controle
NFE10 já existente, depois6 e4. Só um resultado com RTF p50<1, três falas válidas,
sem falhas/silêncio ou indício de limite AR, permite planejar a matriz dos sete
casos, streaming nativo e ASR/conteúdo. A detecção de limite não prova EOS
semântico: uma fala curta por omissão também pode ser rápida. Conservar os WAVs
e rejeitar regressões de conteúdo/pronúncia/caráter robótico na comparação comum.
Nenhum vencedor é escolhido só pelo gate de desempenho.

`--matmul-precision high` é outra ablação, inicialmente com NFE10; `highest` é
o controle medido. O setter ocorre antes de carregar/otimizar o modelo e os
flags reais TF32/CUDNN são gravados. O backbone BF16 já usa baixa precisão;
efeitos tendem a atingir operações FP32/AudioVAE. Não misturar NFE menor e TF32
no primeiro contraste, nem prometer equivalência numérica ou de timbre.
O plano privado com comandos e hashes do controle está em
`voxcpm2/nfe-ablation-plan.json`. Nada foi colocado na fila GPU.

## Otimização oficial e comparação proposta

No [pin oficial](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/model/voxcpm2.py),
`model.optimize()` aplica `torch.compile(mode="reduce-overhead",fullgraph=True)`
a quatro alvos: os dois `forward_step` dos LMs, o encoder local e o estimador
DiT. O device deve ser exatamente `cuda`; `cuda:0` não passa nessa condição.
O método captura exceções e pode devolver o modelo parcialmente otimizado.
O novo runner exige os quatro wrappers, kernels gerados e recibos de execução;
uma advertência não vira sucesso compilado.

O modo reduce-overhead do Torch2.6 solicita CUDAGraphs, mas um wrapper compilado
ou nó gravado não comprova replay. O probe instrumenta temporariamente
`torch.cuda.CUDAGraph.replay` nos três primeiros calls e restaura o método antes
das repetições cronometradas. Registra replays observados, nós gravados, kernels
e contadores/avisos separadamente. Só apresenta CUDAGraph como verificado se o
hook realmente observar replay. Falha de compilação gera `failure.json` com
stack trace; não executa fallback eager sob o nome de compilado.

A primeira comparação será **base sem módulos LoRA**, eager e oficial compilada,
em processos novos e GPU isolada. No
[LoRALinear oficial](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/modules/layers/lora.py),
desabilitar `scaling` não elimina os dois GEMMs extras; removê-los da topologia
evita atribuir esse custo à Base. A segunda comparação, se a primeira justificar,
usa o mesmo checkpoint de adaptador: LoRA eager, fundido eager e fundido compilado.
A fusão é um adaptador local pela fórmula `W + scaling·B@A`, não uma funcionalidade
oficial encontrada nesse pin.

No controle usou-se NFE10, CFG2, seed40733133, ausência de denoise/normalização/
retry e processamento final24k. Qualquer NFE menor é uma ablação de qualidade,
conforme os gates acima. O primeiro call de cada caso fica separado; outros
dois fazem warmup/captura antes da matriz. Depois há no mínimo três repetições por texto, com custo por chamada,
contagem de passos AR, RTF e pico VRAM. O custo médio por passo inclui prefill,
DiT e VAE e é rotulado assim. Agora o smoke padrão usa somente `v_reply`;
antes de escolher modelo, repetir os sete casos comuns e verificar áudio/ASR.
Não abre o teste final para escolher a configuração.

O probe mede WAV completo com resampling/PCM/escrita incluídos, sem HTTP. Não
atribui streaming ou TTFA de cliente a essa medição. O streaming nativo existe
na fonte e será uma etapa separada se os custos aquecidos forem promissores.

## Fusão BF16: resultado CPU e limitação real

`training/v2_voxcpm_optimized_probe.py --action cpu-check` executou fusões reais
de matrizes pequenas em CPU, sem inicializar CUDA. Em FP32, LoRA zero foi exata
e nãozero teve erro máximo2,38e-7. Isso não autoriza prometer equivalência para
pesos2B BF16.

No teste solicitado com **peso BF16, A/B FP32 e referência sob autocast BF16**:

| Intensidade do B | Erro máximo saída | Erro relativo L2 | Updates que desapareceram ao arredondar W fundido |
| --- | --- | --- | --- |
| std0,01 | 0,00390625 | 0,002629 | 1 de35 elementos nãozero |
| std1e-5 | 0,00003052 | 0,000008587 | 35 de35; norma retida0 |

A fusão pode apagar atualizações pequenas mesmo quando a diferença de saída
parece pequena. É uma nova rota de inferência: exige recarregar os pesos
selecionados e comparar textos/áudio/ASR comuns, junto à identidade, sem aceitar
o piloto de quatro updates como evidência de qualidade. Recibo com métricas,
dtypes, fórmula e hashes em
`/var/tmp/aegis-v2-work/voxcpm2/optimized-preparation/cpu-check.json`.

## Toolchain e orçamento preparados

Ambiente reaproveitado: Torch2.6.0+cu118, Triton3.2.0, Python3.12.3, GCC13.3,
driver535.309.01. Headers estavam ausentes. A cópia somente leitura do container
parado revelou3.12.13 e foi descartada; o container permaneceu `exited`, sem
modificações. Extraíram-se apenas215 headers do pacote exato
`libpython3.12-dev_3.12.3-1ubuntu0.17_amd64.deb`, SHA256
`c123ddab7763e45199e21b4ac8545e705f20a0315b9ada4c236b52fb088c4e16`.
O download5,68MB foi apagado depois; o overlay completo ocupa1,92MB.

Imports de Inductor/Triton, uma extensão mínima com Python.h e o `cuda_utils`
real do Triton compilaram/importaram em CPU, sem chamar métodos do driver.
`torch.cuda.is_initialized()` permaneceu falso antes/depois. Recibo e ambiente
privado em `voxcpm2/compile-prereqs.json` e `compile-headers/child-env.json`.
CPATH vale só no subprocesso do probe; não houve nova instalação/ambiente.

O `ptxas` empacotado com Triton é **12.4.99**, apesar do Torchcu118. O driver535
supera o mínimo publicado para CUDA12 minor compatibility, mas há limitações
de PTX/JIT e funcionalidades de driver; o teste CPU não resolve essa pergunta.
A compatibilidade de kernels/CUDAGraphs será verificada na A2000 exclusiva.
Fonte: [compatibilidade NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Todos os caches e temporários de compilação, extensões e cache do driver ficam
em `voxcpm2/compile-cache`, com um compilador/thread e teto512MiB. Pico adicional
estimado1GiB antes do início. O runner faz contagem física global com hardlinks
deduplicados e recusa iniciar se não couber no teto40GB ou faltar5GiB de reserva
mais a margem. Um watchdog local verifica o cap, reserva e total; em violação,
registra `safety-stop.json` e sinaliza **somente o próprio PID do probe**, sem
atingir fila, treino ou grupo de processos. Não duplica pesos completos.

## Comandos preparados para uma janela posterior

O modo padrão é plano, sem Torch/GPU. O modo CPU-check provado é:

```bash
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_voxcpm_optimized_probe --action cpu-check
```

Os comandos abaixo só podem ser filhos da fila GPU privada, depois de parar
o backend anterior com checkpoint íntegro e confirmar que todos os ASRs estão
suspensos pelo arquivo `pause-asr`. O guard recusa outro processo CUDA ou ASR
CPU ativo; não para nem reconfigura outro backend.

```text
python -m training.v2_voxcpm_optimized_probe --action execute --gpu-authorized --topology base --run /var/tmp/aegis-v2-work/voxcpm2/base-eager-controlled
python -m training.v2_voxcpm_optimized_probe --action execute --gpu-authorized --topology base --optimize --run /var/tmp/aegis-v2-work/voxcpm2/base-compiled-controlled
python -m training.v2_voxcpm_optimized_probe --action execute --gpu-authorized --topology lora --checkpoint /var/tmp/aegis-v2-work/voxcpm2/pilot-gpu/step-0004.pt --run /var/tmp/aegis-v2-work/voxcpm2/adapter-eager-controlled
python -m training.v2_voxcpm_optimized_probe --action execute --gpu-authorized --topology merged --optimize --checkpoint /var/tmp/aegis-v2-work/voxcpm2/pilot-gpu/step-0004.pt --run /var/tmp/aegis-v2-work/voxcpm2/adapter-merged-compiled-controlled
```

Usar explicitamente o Python do ambiente F5 nos jobs reais. Cada processo guarda
preflight, receita/hashes, warnings, provas de residência integral dos parâmetros,
buffers e KV na GPU, métricas/áudio por chamada, summary ou failure. Os caches
podem ser reutilizados, registrando se estavam presentes. Um ganho por compilação
ou remoção de LoRA continua condicionado à mesma clareza PT-BR e ao caráter
robótico; nenhuma avaliação auditiva foi inventada nesta preparação.
