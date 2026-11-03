# Monitor local do estágio de idioma VITS

Agenda inicial do supervisor: revisão às07:12UTC e primeira época estimada
por volta de07:22UTC em03/11/2026. São horários de revisão/previsão, não prazo
de interrupção do treino. Cada época completa dispara ASR local automaticamente;
o gate de idioma precede qualquer decisão de especialização ou identidade.

`training/v2_language_monitor.py` acompanha somente o treino em
`/var/tmp/aegis-v2-work/vits/language`. Verifica progresso/processo/fila a cada
30 segundos, grava heartbeat a cada30 minutos e mede o orçamento físico a
cada5 minutos. Não usa GPU nem altera treino, pesos ou dados. Não abre o teste
final. Falhas, NaN, falta de progresso por10 minutos, reserva abaixo de5GiB,
orçamento de trabalho acima de40GB e novas épocas geram `review-needed.json`
na raiz do treino para revisão humana/agente.

Para `probes/initial/results.json` e cada `probes/epoch-*/results.json`, exige os
sete IDs de validação e verifica hashes dos WAVs antes de criar snapshots por
hardlink. A troca atômica ou remoção dos arquivos de origem não muda o inode
preservado para a leitura do ASR. O fingerprint inclui manifesto, WAVs, textos
fixos de validação e código do scorer; o mesmo conjunto não é executado duas
vezes. Jobs, recibos e scores ficam em `asr/{label}-{hash}/`.

O scorer existente `benchmarks/v2/asr_score.py` roda no ambiente F5, CPU int8,
nice19 e dois threads, sem rede e com `CUDA_VISIBLE_DEVICES` vazio. Compara
texto cru, fala esperada fixa e texto efetivo do frontend, conservando os erros
e limitações do reconhecedor. Resultados automáticos não afirmam audição,
identidade vocal nem autorizam parar o treino.

Criar `/var/tmp/aegis-v2-work/pause-asr` impede novos jobs e suspende o grupo do
ASR ativo com SIGSTOP em até30 segundos. Remover o arquivo retoma com SIGCONT.
O monitor não envia esses sinais ao processo de treino. Durante a pausa também
adia a varredura de disco para não adicionar sua carga aos benchmarks de produto.

Retém seis snapshots de época recentes, o inicial, a referência de menor WER
ASR e a época correspondente ao melhor checkpoint de validação do treinador.
Essas proteções são retenção de evidência; o ASR não seleciona o modelo vencedor.
Um snapshot em avaliação permanece protegido. Histórico de métricas e hashes é
compacto; o log do monitor usa rotação de1MiB com uma cópia anterior.

O orçamento inclui todas as raízes `/var/tmp/aegis*`, o ZIP de origem, o worktree
experimental e, conservadoramente, todo `~/.cache`. Conta blocos alocados,
deduplicando hardlinks globalmente e sem seguir symlinks. Exclui os modelos,
caches e perfil preexistentes de produção e a imagem Docker imutável de rollback;
nenhuma imagem nova foi construída. A contagem tem seu próprio inventário, sem
remover dados ou interromper treino por atribuição inconclusiva de cache.

Processo lançado com `Popen(start_new_session=True)`. Estado e retomada:

```text
/var/tmp/aegis-v2-work/vits/language/asr/launcher.json
/var/tmp/aegis-v2-work/vits/language/asr/monitor-recipe.json
/var/tmp/aegis-v2-work/vits/language/asr/status.json
/var/tmp/aegis-v2-work/vits/language/asr/heartbeat.json
/var/tmp/aegis-v2-work/vits/language/asr/budget-latest.json
/var/tmp/aegis-v2-work/vits/language/asr/active.json
/var/tmp/aegis-v2-work/vits/language/asr/state.json
/var/tmp/aegis-v2-work/vits/language/asr/monitor.log
```

PID/start ticks/boot ID e receita do processo estão registrados. Um lock impede
monitores duplicados. Reiniciar o monitor com a mesma receita reconcilia o job
CPU existente por PID/start ticks e recibo persistido, em vez de lançar outro
ASR sobre os mesmos áudios. O primeiro lançamento apresentou uma colisão de
argumento no logger de avisos; a correção e o reinício ficaram registrados no
log e não afetaram o processo de treino.

Validação: quatro testes passaram (deduplicação por hashes/política, retenção,
hardlinks no orçamento e gravação de aviso). O teste de integração confirmou
o processo ASR no estado `T` com o arquivo de pausa e solicitou retomada pela
remoção do mesmo inode do arquivo de teste. Evidência em `asr/pause-test.json`.
