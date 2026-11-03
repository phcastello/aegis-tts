# Recuperação local do FastPitch

O controlador `training/v2_fastpitch_recovery_watch.py` observa o job073 e
sucessores da mesma receita. A versão reforçada foi iniciada em 2026-11-03
às 09:27:48 UTC com PID e PGID 206539 (`start_ticks=370417994`), independente
do terminal. Substituiu somente as versões anteriores do monitor CPU. É CPU puro;
`/proc/206539/maps` não continha `libtorch` nem `libcuda`. O treino 073 continuava
no PID 205461; nenhum retry foi criado durante a instalação.

OOM anterior ocorreu no backward1616 do job072. O job073 retomou o checkpoint1500
com `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` e passou pelo lote que
falhou. Às09:17:50UTC, o checkpoint publicado era2000, epoch10/cursor1630,
SHA256 `748373681388a3e5b971dae42785e617e370f3f2adf91dd63ecf2b63df2ce938`,
melhor validação4.1585188231. Passos em `progress.json` são diagnóstico; nunca
substituem o checkpoint recuperável.

## Política executável

O loop consulta os arquivos locais a cada15s e mantém heartbeat resumido a
cada1800s. Apenas uma execução do controlador pode obter `watcher.lock`.
Não altera o trainer, pesos, dataset, perfis ou serviço de produção.

Um retry exige recibo `done/` com falha/returncode 1 e exceção final CUDA OOM no
log do mesmo job. Saída 0, sinais (inclusive SIGKILL), NaN, erro diverso e progresso parado há10min
produzem decisão sem restart ou `review-needed.json`. O controlador respeita
`finish-gpu` e o veto `/var/tmp/aegis-v2-work/disable-fp-auto-recovery`.

A recuperação passa por8×2 →4×4 →2×8 →1×16, conservando batch efetivo16,
BF16, Adam, scheduler, scaler, RNG, posição dos dados, partições, perdas e
agenda de alinhamento. Mantém o allocator expansível e acrescenta
`--resume-allow-resource-change`. OOM em1×16 exige revisão. Mudanças operacionais
de microbatch são registradas pelo trainer; os parâmetros científicos não
podem mudar silenciosamente.

Antes de publicar qualquer retry, o auditor separado
`training/v2_fastpitch_checkpoint_audit.py` carrega o estado completo em CPU,
verifica SHA do arquivo e recibo, finitude do modelo e Adam, scheduler, scaler,
RNG, cursores, configuração, hashes de dados e fontes. Não restaura RNG em
CUDA nem executa forward. Estado salvo anterior ao último microbatch falho é
preservado. Provas CPU no checkpoint1500 verificaram173 tensores do modelo e501
do otimizador; mudanças8×2→4×4 e8×2→2×8 passaram; alterar `binary_start` foi
rejeitado. A qualificação CPU final do checkpoint 2000 passou todos os 46
checks, com `cuda_initialized=false`; recibo
`recovery-watch/qualification-complete-adam.json`.

O auditor reconstrói a estrutura do modelo somente em CPU para conferir o
mapeamento completo dos 169 parâmetros, a cobertura e os shapes/dtypes dos
momentos Adam de todos os 167 parâmetros ativos. A fonte NVIDIA pinada declara
`attention.attn_proj.weight/bias`, mas nunca usa essa projeção no forward: esses
dois parâmetros sem gradientes são a única exceção explícita à exigência de
momentos. A exceção e a fonte são verificadas; não são inventados momentos para
preencher lacunas. Falta de estado ativo, momento parcial, shape errado ou
parâmetro omitido do grupo são rejeitados.

O controlador só publica jobs na fila serial privada quando ela e a GPU estão
ociosas, o worker exclusivo está vivo e a janela autorizada permanece ativa.
O job contém um pequeno guard que o próprio GPUworker executa: confirma worker
pai, ausência de outro processo GPU, checkpoint/auditoria/receita íntegros e
ausência de retomada posterior pelo operador; depois faz `exec` do trainer no
mesmo PID. O controlador nunca inicia o treino diretamente. Uma retomada manual
posterior ao erro torna o retry antigo obsoleto.

Jobs posteriores com a mesma receita são adotados automaticamente, inclusive
se terminarem entre polls. Depois de uma pausa normal para diagnóstico, basta
ao operador enfileirar sua retomada habitual; não existe flag especial de
rearme. Um job posterior no mesmo output com receita diferente também suprime
retries antigos, porém exige revisão para habilitar recuperação automática da
nova receita. Para rearmá-la explicitamente, encerre somente o PID do monitor
e reinicie o comando abaixo com `--initial-job NOME-DO-NOVO-JOB.json`, existente
em `queue/` ou `done/`. O estado e os tickets anteriores são reconciliados.
Um sinal para parar o monitor durante a auditoria também veta a publicação:
o estado `running` é conferido imediatamente após a auditoria e antes da fila.

Tickets de recuperação possuem nome determinístico pelo SHA do recibo de
falha, escrita atômica e publicação exclusiva. A intenção persistida é
reconciliada com fila e `done/` após queda do monitor; a publicação não cria uma
segunda execução. Há reserva mínima de5GiB antes de recuperar. Nenhum arquivo
grande é copiado para a auditoria.

## Operação e evidências

Estado local privado:
`/var/tmp/aegis-v2-work/fastpitch/language/recovery-watch/`.
Guarda `recipe.json` (PID/start/hash/comando), `status.json`, `state.json`,
`heartbeat.json`, auditorias e tickets. Logs do controlador são limitados a
1MiB + uma cópia; log da auditoria retém cauda limitada; logs GPU continuam sob
o limite do worker. Falhas e decisões ficam em
`/var/tmp/aegis-v2-work/fastpitch/language/review-needed.json`.

Comando do processo destacado (ambiente `CUDA_VISIBLE_DEVICES=''`):

```sh
/usr/bin/python3 -m training.v2_fastpitch_recovery_watch \
  --initial-job 073-fastpitch-allocator-recovery.json
```

Auditoria avulsa, somente CPU:

```sh
CUDA_VISIBLE_DEVICES='' \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_fastpitch_checkpoint_audit \
  --run /var/tmp/aegis-v2-work/fastpitch/language \
  --output /var/tmp/aegis-v2-work/fastpitch/language/checkpoint-audit-manual.json
```

Onze testes CPU passaram, cobrindo evidência final de OOM, parada manual,
invariância da receita, batch efetivo, publicação exclusiva, reconciliação
após crash, supersessão manual, identidade do recibo e mudanças de recursos
versus dados/agenda. Incluem parada durante auditoria com mocks temporários e
checkpoint Adam incompleto ou com shapes errados. Não provocamos OOM nem
submetemos jobs artificiais à fila real. O primeiro retry real, se necessário,
ainda precisa fornecer sua própria evidência de retomada bem-sucedida.

```sh
python3 -m unittest tests.test_v2_fastpitch_recovery_watch \
  tests.test_v2_fastpitch_checkpoint_audit -v
```
