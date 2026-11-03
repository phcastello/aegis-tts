# VoxCPM2: provas HTTP no checkpoint 400

Política validada para a próxima seleção de pesos: KV1024, NFE10/CFG2,
ramos LoRA r8 BF16 sem merge, AudioVAE FP32, partes neurais na A2000,
FIR41 Kaiser10 de48→24k, `ptbr-numbers-v1`, passthrough explícito e reserva
inicial de320 ms. O checkpoint400 é um controle de engenharia; não é a
seleção final da voz. O treino continua e os pesos finais salvos terão nova
medição. O perfil protegido Qwen/v1 não mudou.

Foram seis execuções, sete frases de validação×3 cada:126 respostas completas.
O serviço aqueceu uma frase fora do corpus três vezes na mesma thread
persistente. Nenhum caso recebeu aquecimento próprio. Rep0 é texto inédito
após readiness; rep1/2 são repetições. Concorrência1, ASR/treinos concorrentes
pausados, PCM16LE mono24k e mesmos textos/onset rule. O teste final de seis
frases não foi aberto.

| Caminho | Reserva | TTFA p50 da matriz | Faixa dos p50 por frase | TTFA máximo | RTF longo p50 | Lacunas>10ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, checks síncronos antigos | 0 | 1160,2 ms | 1134,4–1188,6 ms | 1263,1 ms | 0,944 | 3 em3 requests;2747,5 ms |
| Frontend `f5-refinement-1`, checks antigos | 0 | 1140,8 ms | 1114,4–1164,1 ms | 1207,6 ms | 0,937 | 3 em3;2135,5 ms |
| Original +CacheGuard | 0 | 267,4 ms | 247,0–288,6 ms | 538,4 ms | 0,723 | 2 em1;180,5 ms |
| Números +CacheGuard | 0 | 268,0 ms | 245,8–287,2 ms | 311,7 ms | 0,721 | 5 em3;305,4 ms |
| Original +CacheGuard | 320 ms | 594,4 ms | 569,5–607,0 ms | 1097,5 ms | 0,723 | 0/21 |
| Números +CacheGuard | 320 ms | **584,4 ms** | **563,9–608,5 ms** | 975,4 ms | **0,717** | **0/21** |

TTFA é o proxy digital do cliente HTTP: envio até reprodução FIFO da fala
detectada por RMS20ms/−45dBFS/duas janelas, incluindo a reserva320ms. Exclui
o atraso do dispositivo físico. Há três observações por frase: não se estima
p95 por caso; os máximos mostram a variabilidade. Zero lacunas cobre as42
respostas com reserva desta prova, não qualquer texto/carga futura.
Todas as linhas usam VoxCPM2/LoRA400; `f5-refinement-1` identifica somente a
receita de texto aplicada antes do modelo.

No numérico+320, o primeiro PCM válido teve p50 de564,6ms; a primeira fala
confirmada já contida no PCM também564,6ms. O FIFO acrescentou o offset
audível, chegando a584,4ms. Incluem-se frontend, segmentação, modelo, VAE,
FIR, quantização, buffer e HTTP. Tempos anteriores do motor nativo com três
warmups por caso não são latência deste gateway.

O [diagnóstico de filesystem](voxcpm-cacheguard-20261103.md) explicou cerca
de1,7s/segmento de custo removível. Antes/depois do CacheGuard, WAV e PCM da
rep0 original foram bit a bit iguais nos sete casos. Antes/depois da reserva,
os sete WAV/PCM permaneceram bit a bit iguais em **cada** frontend. A reserva
alterou a entrega; não corrigiu ou normalizou áudio. A configuração sem reserva
não é a escolhida para interação robusta diante das lacunas observadas.

Partida de processo e readiness ficam separadas das solicitações aquecidas:

| Execução com320ms | Processo→ready | Carga do backend | Inicialização interna, incluindo warmups | Pico VRAM startup / matriz |
| --- | ---: | ---: | ---: | ---: |
| Original | 167,58 s | 67,00 s | 165,12 s | 6084 /6086 MiB |
| Números | 167,52 s | 66,34 s | 164,97 s | 6084 /6086 MiB |

A prontidão foi consultada a cada1s; o backend mede internamente sua própria
inicialização. Antes de ready, solicitações retornam503. A carga inclui
verificação dos pesos e materialização CPU→GPU; compilação/captura e três
sínteses precedem readiness. É partida fria de processo com cache de compilação
existente, não uma requisição fria de fala bem-sucedida. Não se limpou o cache
para fabricar outro cenário.

Nas provas com320ms, o pico de alocação PyTorch no startup foi5382,6MiB e o
pico reservado5642MiB. O processo atingiu cerca de10,323GiB de RSS na carga.
O sampler nvidia-smi opera a500ms e pode perder picos breves; seus6086MiB
incluem contexto/driver, enquanto o contador PyTorch é do allocator. Replays
reais de CUDA Graph foram contados:13308 no original e13452 no numérico,
incluindo warmups e matriz. Wrappers presentes não foram usados como prova.

Ambas as provas finais tiveram21/21 estados `completed`, WAV completo mono24k,
PCM streaming real, DELETE depois de4096 bytes PCM válidos, estado final
`cancelled`, nova síntese após cancelamento e health/ready/queue200. Fontes e
checkpoint permaneceram íntegros; os processos8002 foram encerrados pelo
executor. O operador retomou o treino. Nenhum servidor experimental ficou ativo.

Recibos sob `/var/tmp/aegis-v2-work/voxcpm-gateway`:

- `preparation/gateway-engineering-execution-summary.json`: métricas por frase,
  recursos, contratos, hashes dos recibos e condições das seis provas.
- `cacheguard-original400-startup320/` e `cacheguard-numbers400-startup320/`:
  resultados HTTP, startup, GPU sampling, SSE/timers, contratos e WAVs rep0.
- `cacheguard-audio-equivalence.json` e `startup320-audio-equivalence.json`:
  comparações exatas de WAV/PCM, sem avaliação auditiva inventada.

Checkpoint comum: `selections/aegis-step400.pt`,54741414 bytes,
SHA256 `1c9409ddf9e39352618984221d3334e607e3168275cc7585e466d23c409e8c0e`.
Fonte startup320: commit `4c75cae`; runtime
`76b9ea179e290a2fa16f6e85947d021042a407cecb551a6f7f87bf12caa6395c`,
runner `c27b5ba924d2aeb3a2d2d789ef502d8724173054bc626fd251bef8ec5a38d7bc`.
O comando testado está nos manifests e em `NEXT-voxcpm-startup320-probe.json`;
exige GPU worker exclusivo, GPU/CPU isoladas e diretório novo, com
`--frontend ptbr-numbers-v1 --startup-ms 320`.

A decisão congela a **política de inferência**, não aprova identidade nem pesos
finais. ASR é auxiliar e estes helpers não fizeram escuta. Não há promoção ao
perfil protegido. A medição dos pesos selecionados e o teste final reservado
continuam como etapas do encerramento do treinamento.
