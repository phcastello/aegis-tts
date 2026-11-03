# Supervisor CPU da especialização VoxCPM2

`training/v2_voxcpm_quality_monitor.py` está ativo para o run privado
`/var/tmp/aegis-v2-work/voxcpm2/aegis-lora`, lançado pelo responsável em
03/11/2026 às 12:47:44 UTC e substituído após revisão às 15:26:41 UTC,
PID atual 232984/start 372571346. O processo anterior foi encerrado pelo
responsável quando não havia ASR ativo; o trainer não recebeu sinais.
Não carrega Torch/modelos nem usa ou reinicia a GPU. Reutiliza os utilitários
de arquivos, identidade de processo e orçamento dos monitores anteriores.

O monitor observa `validation.jsonl` a cada 30 s e congela os sete WAVs PCM
já renderizados pelo treino. Hardlinks conservam os inodes mesmo depois da
retenção do trainer. Cada snapshot guarda os bytes de `results.json` e
`asr-jobs.json`, hashes das fontes/WAVs, jobs ASR próprios, textos fixos
verbatim/spoken/effective e a política de avaliação. São somente os sete
textos comuns de validação; o teste reservado não é aberto.

NFE esperado padrão é 10 e fica explícito na política; outro valor exige
`--expected-nfe` correspondente. Resultados de NFE diferente são recusados,
sem reescrever rótulos. O áudio é eager, sem DSP extra e sem equivalência
presumida com o streaming compilado. Não é benchmark de produto.

O ASR é o scorer local existente, pinado por hash, com recibo do modelo
também fixado. Roda no ambiente F5, CPU/int8, duas threads, nice19, sem rede
e com CUDA oculta. A cada segundo o monitor verifica `pause-asr`; só envia
STOP/CONT ao grupo que comprova boot, UID, PID/start ticks, sessão e os
argumentos exatos do scorer/jobs. Ao encerrar, pausa incondicionalmente esse
filho verificado, persiste sua identidade e permite reanexação na próxima
instância. Nenhum sinal vai ao trainer.

O watchdog permite 1.800 s de execução ativa do ASR, excluindo a pausa, e
mata apenas o grupo verificado ao exceder esse limite. O deadline interno
do scorer foi alongado para não contar SIGSTOP como trabalho: a garantia de
1.800 s ativos depende deste monitor vivo. Se o monitor sofrer SIGKILL ou
falhar antes do finally, a instância seguinte deve reanexar o processo; não
há um segundo daemon. Logs têm limite de 1 MiB, com uma cópia anterior.

Uma observação só é comparável quando existem sete resultados ASR únicos,
completos e bem sucedidos, hashes corretos e política idêntica. Uma falha de
síntese/ASR permanece no denominador de sete e impede WER parcial de virar
best. `metrics.json` conserva essas contagens, `product:false` e os WERs
verbatim/spoken/effective quando completos. `comparison-to-initial.json`
explicita antes/depois, `filtered:false`, a mesma política e a ausência de
comparação quando o inicial ou o atual ficou incompleto. São avisos para
revisão de áudio e conteúdo, sem aprovação automática da voz.

A captura do checkpoint ocorre separadamente porque o trainer escreve a
validação antes de publicar os pesos. Só há link quando o sidecar de last
ou best informa o mesmo passo e o hash exato do inode congelado confere.
A captura pode continuar depois de terminar o ASR. Publicação incompleta
fica pendente; não se associa checkpoint posterior ao áudio anterior.
Um link órfão de queda antes do commit do estado é reconciliado ou removido
somente na área do monitor. Não há desserialização do modelo no supervisor.

São conservados os três checkpoints exatos de validação mais recentes,
independentemente da posição no ASR, além do melhor estado ASR preservado.
Cada arquivo deve ter até 70 MiB; o checkpoint medido existente é
aproximadamente 54,7 MB. Um resultado novo ainda sem pesos não expulsa os
três estados já vinculados. O melhor recuperável anterior continua retido
até existir o checkpoint do novo melhor. Se os receipts já avançaram e aquela versão
não pode ser vinculada, isso gera aviso explícito. Pesos de tentativas
encerradas sem uso futuro são removidos, preservando `last.pt`/`best.pt` do
trainer. Áudios mantêm seis avaliações recentes, inicial, best ASR, best do
treino, ativo e eventual estado anterior ainda necessário à recuperação.
A limpeza verifica proprietário, caminho real e uso com fuser.

O orçamento conserva a definição do laboratório: todos os `/var/tmp/aegis*`,
ZIP de origem, worktree e cache pessoal contabilizado conservadoramente,
com hardlinks deduplicados; produção e imagem imutável de rollback ficam
fora. Reserva estimada adicional: 384 MiB, além de 5 GiB físicos livres.
Orçamento inconclusivo/ultrapassado bloqueia ASR/snapshots novos e gera
aviso; não apaga insumos nem para o treino. Idade de progresso acima de
10 min, NaN/OOM/traceback, saída do processo e problemas de espaço também
geram revisão. Heartbeat resumido a cada 30 min; não há chamadas à IA.

## Operação após revisão do responsável

Receita para uma sessão independente do terminal, usada no lançamento:

```sh
cd /home/phc/aegis-v2-lab
/usr/bin/python3 - <<'PY'
import os, subprocess
command = ['/usr/bin/python3', '-m', 'training.v2_voxcpm_quality_monitor',
           '--run', '/var/tmp/aegis-v2-work/voxcpm2/aegis-lora',
           '--expected-nfe', '10']
process = subprocess.Popen(command, cwd='/home/phc/aegis-v2-lab',
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
print(process.pid)
PY
```

O lock impede duas instâncias. `quality-monitor/recipe.json` registra PID,
start ticks, boot, comando, hashes, limites e cadências. Para reanexar após
uma queda, repetir o comando; ele identifica o filho anterior antes de
enviar qualquer sinal. `--once` observa e congela sem lançar ASR novo.

Validação inicial: sete testes stdlib próprios passaram;
com os utilitários compartilhados, 19 passaram em 0,354 s. A revisão do vínculo
gate→exec acrescentou três testes; os 10 próprios passaram em 0,335 s e foram
repetidos pelo responsável. Exercitam retenção
do trainer após congelamento, publicação tardia e link órfão do checkpoint,
captura depois de ASR, sete casos incluindo falha, pausa durante preparação
do lançamento, sinais verificados e parada. O supervisor continua sem
carregar modelos ou usar GPU; somente seu filho executa o ASR CPU.

## Prova do lançamento e primeira captura

Fonte lançada SHA256
`12a6b1298afc729880681f8517ab945f105d7526c863a42fbf5811c1145b56b5`.
O worker conserva o comando do gate `advance-voxcpm-aegis.py`, mas o mesmo
PID 218142 já executa o trainer. A saúde agora lê `/proc` e verifica UID 1002,
start 371534732, PPID 195559 e worker start 369248883, ancorado no snapshot da
operação. O comando real e o comando original do gate ficam registrados,
permitindo associar corretamente o log e a saída. `progress.json` é o
arquivo efetivo; antes de existir, o status distingue a inicialização ou
primeira validação. Isso é evidência de saúde, sem autorizar ações na GPU.

A inspeção em leitura confirmou o processo desacoplado e ausência de
`libtorch`/`libcuda` no monitor. A observação inicial congelou sete WAVs NFE 10
e o checkpoint exato 0 por hardlink, com SHA256
`f97a2d096aee132d1cd612c0b4c2fd2fcfc851ea6981dff7814fae329c16c21b`,
concordante com o sidecar e o inode do best inicial do treino.

O filho ASR 219093 executou com duas threads/nice19 e encerrou 7/7 casos.
Inicial **eager** do adapter 0: 22/135 edições, WER 16,30%. O baseline nativo
**compilado** separado tinha 15/135, WER 11,11%. São condições diferentes;
o antes/depois do treino deve usar o inicial eager desta mesma política,
sem atribuir a diferença ao treino ou inferir identidade vocal.

Recibos privados: `voxcpm2/quality-monitor-launched.json` e
`voxcpm2/aegis-lora/quality-monitor/launch-verification.json`, sob
`/var/tmp/aegis-v2-work/`. Próxima revisão do responsável: 13:10 UTC ou erro
antecipado, conferindo áudio/ASR do passo 200 e seu checkpoint de mesmo
passo/SHA. Nenhuma consulta à IA por minuto é necessária.

## Retenção de três checkpoints: substituição verificada

O responsável aplicou a fonte revisada SHA256
`ea4deae146d762d9793a71889762a931b084c9a7be1462f05658d8272be1a818`
e repetiu 15 testes stdlib antes de lançar a substituição. Os cinco casos
novos cobrem três estados recentes com ASR pior, remoção apenas do link
próprio antigo, preservação por fuser, recusa de UID/path e captura tardia
de uma validação concluída que não é best. Os dez contratos anteriores,
incluindo sete casos e passo/SHA exatos, também passaram.

A conferência leve das 15:38 UTC vinculou PID/start/UID/comando ao recibo
novo e verificou os checkpoints 800, 1.000 e 1.200 por sidecars e inodes,
sem reler pesos grandes. O passo 1.000 foi capturado do best exato, SHA
`3a8c006a0ff980d78f29d943468bdf2dca6801d9e1e2f60ce4a023da583c06e3`;
seu ASR 16/135 não o promoveu e não causou a remoção. O passo 1.200 teve
ASR 13/135, WER 9,63%, sob a mesma política eager NFE 10. Essa é evidência
de conteúdo, sem aprovação da identidade robótica. A prova está em
`/var/tmp/aegis-v2-work/voxcpm2/quality-monitor-retention3-launch-proof.json`.
