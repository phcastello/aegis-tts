# Preparação HTTP dos checkpoints1800 e2800

Dois planos CPU estão prontos para confirmar os checkpoints salvos no gateway
real. Ambos têm12/135 edições no ASR eager já existente; ainda precisam de
recarregamento compilado, medição HTTP e contratos completos antes da seleção.
Nenhum serviço, GPU, fila, sinal de treino ou novo ASR foi executado nesta
preparação. O teste final continua fechado.

| Plano | Pesos salvos | Output novo |
| --- | --- | --- |
| 095q proposto | passo1800, SHA808fc434… | selection-step1800-numbers-startup320 |
| 095r proposto | passo2800, SHAb60e4d7e… | selection-step2800-numbers-startup320 |

Os dois `--action plan` passaram no Python F5 sem importar Torch; as configs
passaram no parser real. A política é a já comprovada no controle CP400:
ptbr-numbers-v1/reserva320ms, KV1024/NFE10/CFG2/seed fixo, LoRA BF16 sem fusão,
VAE FP32, modelo inteiro na GPU e FIR48→24/passthrough. Esta rodada usa
nenhum prompt fixo. Cache512MiB e overlay existentes são compartilhados.

O código imutável deriva o perfil/candidate HTTP da receita. Seus valores
serão iguais nos dois runs e continuam distintos do perfil protegido v1.
Passo+adapterSHA+configSHA+diretório e IDs próprios no índice de comparação
identificam os pesos; as etiquetas HTTP iguais não implicam mesmo modelo.
Não se deve agregar resultados ASR desses dois checkpoints só por candidate.

Receitas privadas e comandos determinísticos, ainda **não executados**:
`/var/tmp/aegis-v2-work/voxcpm-gateway/selection-preparation-1800-2800/PREPARATION.md`,
com `095q.proposed.json`, `095r.proposed.json`, índice, preflight e recibo.
As configs/planos ficam nos respectivos outputs acima, sob o mesmo diretório
`voxcpm-gateway`. As propostas não foram publicadas na fila.

O pico conservador observado é38,728GB para teto40GB, incluindo margem de
1GiB do segundo run,512MiB do primeiro retido e128MiB de sobreposição de
checkpoint. Há27,281GB físicos livres; o probe refaz o preflight na janela.
As fontes do gateway/runtime/probe mantêm os hashes das provas CP400.

O operador fará a parada suave/exclusividade e sequenciará os jobs. Cada
probe inicia e encerra seu próprio uvicorn8002, mede cold readiness e sete
frases n3 após três warmups fora do corpus, sem prewarm por caso. Confere
WAV/PCM/cancelamento/próxima requisição/saúde e guarda jobs para ASR posterior.
Os resultados desses controles ainda estão indisponíveis; o treino e a
avaliação reservada não foram usados para antecipar a decisão.
