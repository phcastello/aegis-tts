# Custo do gateway e controle numérico — 2026-11-03

Os dois probes HTTP com LoRA400 passaram pelos contratos e encerraram seus
servidores, mas expuseram atraso fixo entre o motor e o áudio entregue. O
diagnóstico e a correção abaixo mantêm checkpoint, modelo, BF16/FP32, KV1024,
NFE10, seed, FIR e DSP. Ganho HTTP da correção ainda precisa ser confirmado.

O caminho antigo chamava `cache_budget` duas vezes antes do primeiro `next`
nativo e duas vezes no reset anterior ao EOF. Medição stdlib, sem modelo ou
Torch, na árvore real de 10.838 arquivos/194.711.552 bytes alocados:

| Parcela | Observações | p50 | máximo |
| --- | ---: | ---: | ---: |
| cache_budget completo | 12 | 430,1 ms | 453,0 ms |
| observe com getters GPU substituídos | 12 | 424,9 ms | — |
| padrão das quatro checagens | 7 | 1.700,3 ms | 2.085,3 ms |
| walk/stat isolado | 5 | 440,1 ms | 447,9 ms |
| statvfs/disk_usage isolado | 5 | 0,035 ms | 0,047 ms |

Isso localiza um custo de cerca de0,86s antes e0,85s depois da geração, sem
atribuir ao modelo a regressão do gateway. A medição foi nice19 com treino/ASR
concorrentes; não substitui o benchmark HTTP isolado. Recibos privados:
`voxcpm-gateway/preparation/cache-cost-cpu.json` e
`cache-cost-components-cpu.json`, sob `/var/tmp/aegis-v2-work`.

`CacheGuard` agora possui uma única thread com prioridade baixa para percorrer
o cache. Cadência5s, limite512MiB, reserva de5GiB, idade máxima do snapshot15s.
Todas as requisições consultam apenas snapshot e flag em memória. Exceção,
thread morta ou snapshot velho bloqueia geração/readiness. Detalhes da falha
ficam no log privado; `/metrics` expõe erro, idade, duração e contagem dos checks.
Há check inicial e um check solicitado/aguardado depois dos três warmups,
antes de readiness. Esse último exige scan iniciado depois do pedido; não
aceita uma varredura já em andamento durante a compilação.

A fiscalização é amostrada: pode haver crescimento entre duas verificações.
Mantêm-se a margem global de1GiB, o teto de40GB e a reserva por volume. O
monitor é fechado e aguardado em `finally`, inclusive após falhas de init/reset.
Cancelar/fechar o gerador e zerar/sincronizar KV continuam na thread CUDA; o
monitor não importa Torch. O runner lê o contador do runtime, sem outro crawl
do cache, e verifica somente saída, volume e margem do preflight.

Medição stdlib posterior do mesmo padrão via snapshots: p50 0,273ms, máximo
0,459ms. Dois checks sincronizados no startup custaram0,856s fora da requisição.
Getters GPU foram substituídos; os tempos não são TTFA. Receita reproduzível:

```bash
python3 -m training.v2_voxcpm_cache_cost
```

Timers leves por segmento registram preflight, primeiro `next`, soma dos `next`,
close nativo, reset e espera na bridge. Não há log por bloco, mudança de áudio
ou chamadas adicionais de aquecimento. O runtime header identifica
`cacheguard5s-age15s`.

O controle `ptbr-numbers-v1` aplica o mesmo helper F5 com `lexicon={}`. Expande
números, horários, datas e unidades; preserva literalmente Aegis, CPU, API e
APIs. O oráculo `14h05` resulta em `quatorze horas e cinco minutos`. Config,
perfil `PtbrNumbers1`, header e esperado ASR efetivo distinguem esse controle de
`original-nfc` e `f5-refinement-1`, que continuam disponíveis. Original segue
o padrão; treino e léxico protegido não mudaram.

Motivação: o ASR HTTP do checkpoint400 reportou14/135 no original e27/135 na
expansão F5 completa; esta piorou nominalmente a comparação e não é a escolha.
O original teve horários ambíguos no reconhecimento, enquanto a expansão
numérica expressou horas/minutos. Isso justifica um controle menor, sem tratar
a transcrição ASR como avaliação auditiva confirmada.

Próximo gate: mesmo LoRA400 congelado, `original-nfc` e `ptbr-numbers-v1` em
runs novos, sete validações×3, somente três warmups de uma frase fora do corpus.
Comparar primeira frase inédita após readiness, repetidas, gaps/RTF e contratos.
O teste final continua fechado. Nenhum resultado desta nota aprova identidade.
