# Limpeza de estados históricos encerrados

Foram liberados1.071.652.864bytes na medição global, mantendo os insumos
originais, referências Qwen/F5 e checkpoints ainda candidatos. O projeto
passou de38.000.500.736 para36.928.847.872bytes alocados. A diferença inclui
os pequenos recibos criados e eventual atividade concorrente do treino.

| Conjunto encerrado | Bytes alocados removidos |
| --- | ---: |
| FastPitch público7000, comparação já medida e não escolhida | 549.629.952 |
| Vox pilotos1–4 e longest last4/best0 | 291.930.112 |
| Vox checkpoints históricos200/1000 | 109.486.080 |
| Gerador diagnóstico VITS da configuração encerrada | 108.544.000 |
| **Total de checkpoints** | **1.059.590.144** |
| WAVs idênticos dos ensaios HTTP400, unidos por hardlink | 12.079.104 |

Dez inodes de checkpoints foram removidos. Cada arquivo foi novamente
conferido por caminho real, UID1002, dev/inode/nlink1, SHA e ausência de uso
por `fuser`. A fila atual não os referenciava. A configuração VITS já estava
formalmente encerrada; eventual tentativa com LR menor começa da base
original preservada. As amostras, métricas, receitas e hashes históricos
permanecem. Esses pesos antigos não permitem mais replay direto; não existe
backup remoto verificado deles. Não se altera a história dos recibos selados.

O primeiro preflight de limpeza usou uma opção `--` que este `fuser` não
aceita e parou antes de qualquer alteração. A execução concluída verificou
os caminhos absolutos como argumentos NAME, com retorno1 e saídas vazias.

Nos dez grupos de WAVs duplicados, todos os caminhos e SHA permaneceram.
Só inode/mtime passam a ser compartilhados. Nenhum áudio foi reamostrado,
normalizado, recortado ou recodificado. A busca ficou restrita aos seis
ensaios HTTP400 encerrados; corpus, originais, referências e escuta1400
ficaram fora do escopo.

FP last7418 e best/selected5500 foram conferidos por inode e SHA depois.
Os dois nomes de5500 continuam no mesmo inode; apagar só um não liberaria
bytes. Vox400/800/1400/1800/2800 e estados atuais seguem mantidos. As análises
200/1000 conservaram seus WAVs/embeddings/metrics, com a indisponibilidade
posterior dos pesos registrada separadamente neste documento.

## Pico da próxima etapa

A fila futura do FastPitch é agora096b, com um preflight de orçamento antes
da execução do comando científico original. O treinador e sua receita não
mudaram. O gate exige GPU livre,40GB de teto e5GiB físicos de reserva, além
de3GB extras conservadores para checkpoint temporário/retidos, probes e logs.
O nome096 foi substituído antes de executar para preservar a identidade dos
jobs que o watcher pode ter registrado.

Após a limpeza, **somente check-only** passou:36.928.876.544+3.000.000.000
=39.928.876.544bytes. O FastPitch não foi lançado; Vox continua na GPU.
A margem contra essa estimativa é pequena e muda com os checkpoints Vox.
O gate será repetido automaticamente no início; não se reduziu a reserva
para fazer a etapa caber. Se não fechar, fará falta nova limpeza/auditoria
antes de iniciar. A recuperação por microbatch conserva eff16 e o gate;
o despacho4×4/2×8/1×16 e a normalização do allocator externo foram conferidos
usando as funções existentes do watcher, sem executar treino.

Recibos privados em `/var/tmp/aegis-v2-work/operations/`:

- `fastpitch-disk-review-20261103T192245.{json,md}`: auditoria inicial somente leitura.
- `cleanup-closed-states-20261103T1930.json`: ações, inodes, hashes antes/depois;
  SHA`99ce74057a193e24cff6902ad9f7a32f825a4af0dfd3fae8937bedd210e66e58`.
- `cleanup-closed-states-20261103.py`: execução com whitelist exata e journal.
- `fastpitch-budget-preflight.py`, `fastpitch-budget-gate-preparation.json`
  e `fastpitch-stage-preflight-check.json`: pico e prova check-only.
- `096-fastpitch-original-job.json`: comando original preservado.

O log agregado continua em `/var/tmp/aegis-v2-work/cleanup.jsonl`.
