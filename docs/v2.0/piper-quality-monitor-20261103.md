# Monitor Piper: implementação para revisão, sem execução real

`training/v2_piper_quality_monitor.py` acrescenta supervisão e retenção privadas
para a nova receita Piper originalcase. Não altera o trainer congelado
`v2_piper_experiment.py` (`2e08e891…`). Nenhum monitor, ASR, treino ou processo
GPU foi lançado nesta implementação. O plano e o lançador privado são propostas.

O processo usa somente a biblioteca padrão. Observa a cada30s, publica
heartbeat a cada1800s e mantém log rotativo de1MiB mais uma cópia. O ASR é o
scorer existente `4ee9f7da…`, Whisper-small fixado em
`536b0662742c02347bc0e980a01041f333bce120`, CPU/int8/2threads/nice19/offline.
Versões, normalização, textos esperados, fontes e hashes são parte da política;
saídas com diferenças nesses vínculos não entram na seleção.

## Publicação e recuperação

O schema Piper é uma lista de sete resultados em `probes/step-*/results.json`.
`last-probes.json`/`best-probes.json` fornece passo e SHA do checkpoint. O
monitor conserva bytes dos manifests, ponteiro e sete WAVs por hardlinks;
confere PCM16/mono/24k, integridade, nome do arquivo, texto, frontend nativo e
ruído0/0. Falhas de geração continuam no denominador7.

Antes do ASR captura exatamente o estado indicado, conferindo SHA, passo,
tamanho, inode e sidecar `last.pt.json` ou `best.pt.json`. O limite é900MiB por
checkpoint. Um estado posterior nunca substitui o indicado. Se já desapareceu,
há registro de indisponibilidade e revisão; não há seleção sem pesos. Um
hardlink interrompido pode ser reconciliado por seu SHA exato. Probes
intermediários encontrados sem ponteiro recebido são registrados como tais.

Há no máximo um checkpoint pendente de ASR e um melhor ASR selecionado. Só uma
execução completa7/7 com WER estritamente menor e checkpoint íntegro substitui o
melhor. Manter o melhor ASR exige pesos completos, além dos last/best por mel
do próprio trainer. Áudios retidos: últimos seis, inicial, melhor ASR, melhor
mel e qualquer execução ativa. Remoção é restrita a arquivos próprios, sem
symlinks, no diretório privado do monitor e sem usuários em `fuser`. Estados
de treino, originais e outros processos não são apagados.

O pico conservador contempla **cinco estados distintos**:
last + melhor mel + melhor ASR + um pendente + temporário de gravação.
Com o estado medido845.928.392B isso representa cerca de4,23GB, antes da
margem. O cálculo usa o maior estado conhecido mais16MiB, limitado ao cap,
desconta apenas inodes já alocados e adiciona256MiB auxiliares. Também exige
orçamento global≤40GB e5GiB livres no volume além da reserva. Uma recusa de
orçamento impede captura/ASR novos e solicita revisão; não descarta o modelo.
O inventário inclui também `/home/phc/aegis-tts/.git`: o worktree isolado usa
esse armazenamento compartilhado, que ficava fora da contagem anterior.
Após essa correção, os54 testes stdlib conjuntos passaram novamente. A função
de pontuação mantém a mesma AST da versão validada com recibos reais; o
monitor continua sem lançamento real.

O vínculo de saúde identifica o processo Piper real após `exec`, com
UID/PPID/PID/start, worker GPU e boot. ASR tem sessão própria e vínculo de
boot/UID/PID/start/jobs/output. `pause-asr` interrompe apenas esse processo;
o estado real do kernel é conferido e `pause-ack.json` só confirma quando há
pausa efetiva ou nenhum ASR vivo. A retomada reconcilia processo existente e
journal de lançamento antes de considerar nova execução. Resultado parcial,
processo morto ou timeout não geram retry automático. A ausência de um handle
do provedor não significa que o processo local morreu.

## Operação proposta

Com o diretório privado da nova execução já criado e a janela GPU conciliada:

```sh
cd /home/phc/aegis-v2-lab
python3 -m training.v2_piper_quality_monitor \
  --run /var/tmp/aegis-v2-work/piper/aegis-originalcase-g1e5-d5e6-v1 \
  --plan
```

O pin padrão é o manifesto originalcase `b212a8eb…`, fingerprint `59568c02…`.
O manifesto lowercase anterior é recusado. Overrides exigem SHA256 explícito
e formam outra receita de geração. O lançador preparado, com `setsid` e
entrada fechada, fica em
`/var/tmp/aegis-v2-work/piper/monitor-preparation-20261103/launch.PROPOSED.sh`.
Ele foi validado por `bash -n`; não foi executado. Lock exclusivo impede dois
monitores sobre a mesma execução. `--once` observa/congela sem lançar ASR.

Validação:32 testes stdlib cobrem publicação concorrente, ponteiro/SHA errados,
retomada de hardlink, inodes/orçamento, cap de pendentes, identidade de
processo, pausa, journal/reattach, ASR parcial/inválido, retenção e limpeza.
Comando: `python3 -m unittest tests.test_v2_piper_quality_monitor -q`.
O import do módulo é testado sem Torch/NumPy/ASR. Plano CLI também executou.
A suíte conjunta de monitores Piper/Vox/FastPitch/VITS passou54 testes. Uma
verificação privada reutilizou três grupos reais de sete saídas do scorer
fixado, conservando hipóteses/settings/versões/métricas, e conferiu o reload
de `source.json` contra sete WAVs históricos do produtor Piper. Passou como
compatibilidade de contrato; não é nova medição de qualidade Piper. Recibo:
`piper/monitor-preparation-20261103/real-scorer-compatibility.private.json`.

## Limites

Não houve teste do monitor com o modelo completo nem lançamento real do ASR;
isso é necessário na próxima janela autorizada. Este monitor não controla
GPU, LR, retomada/OOM ou parada do trainer. Detecta problemas e exige revisão
do supervisor. Seu histórico é limitado512 snapshots; atingir isso bloqueia
novas capturas até manutenção explícita. A gravação de checkpoints completos
e seus estados RNG/otimizador continuam sob responsabilidade do trainer.
ASR não aprova pronúncia ou identidade, e o monitor não mede latência do
produto, não ouve áudio e não acessa teste final.
