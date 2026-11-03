# FastPitch: próxima especialização Aegis preparada por metadados

Hipótese: o FastPitch que qualificar PT-BR no estágio público pode aprender
o timbre robótico em uma etapa Aegis nova. Avançar exige pronúncia inédita e
identidade; loss mel/CTC não aprova a voz. Nenhuma extração, carga de tensor,
GPU ou início de treino foi executado nesta preparação.

A receita foi conferida pelo SHA256
`453c3806b384c0af8aa0fdc1cc79b97282b6940f971cc84ac01ed35c1011705d`.
`training/v2_fastpitch_target_plan.py` produz somente manifests e auditoria
de metadados, sem importar Torch ou abrir áudio, features ou o teste final.
Artefatos privados:
`/var/tmp/aegis-v2-work/fastpitch/aegis-preparation/`.

| Seleção congelada | Falas | Duração | Tokens máximos com bordas | OOV |
| --- | ---: | ---: | ---: | ---: |
| Treino | 1.025 | 13.030,096 s | 214 | 0 |
| Validação | 55 | 722,994 s | 203 | 0 |

Os hashes dos manifests de entrada são `29c81ccc…` e `d3302328…`.
`train.jsonl` e `validation.jsonl` derivados conservam IDs, grupos, áudio
original por caminho/SHA256 e `original_text`. O campo `text` é o original
em NFC; o frontend fixo aplica lowercase e substituições explícitas de
pontuação, sem expandir números ou alterar o nome Aegis. Não há OOV no
VOCAB78 nem sobreposição entre treino/validação por ID, grupo, SHA de áudio
ou texto normalizado. O teste de 50 falas permanece fechado.

O extrator lê o original48kPCM24 mono, reamostra em RAM para24k sem ganho,
corte ou efeitos, e gera mel100/FFT1024/hop256/Hann/padding384/Slaney/log1e-5
compatível com o BigVGAN já fixado. Não reutilizar os mels F5/Vocos: eles
têm outra convenção de frames. A equivalência do extrator BigVGAN e a
reconstrução reservada anteriores estão registradas no journal e na revisão
FastPitch; uma eventual candidata ainda precisa da escuta de identidade.

O cache antigo `fastpitch/target-features` contém **810 treino/180 validação**,
990 objetos e505.511.512 bytes lógicos (507.600.896 alocados). Essa partição
não é a seleção1.025/55. A auditoria leve encontrou950 objetos compatíveis
por ID/texto normalizado/grupo/path/SHA declarado; faltam130, e40 antigos
ficam fora. `reuse-plan.json` lista só os objetos selecionados e seus hashes.
Não foram lidos seus payloads nem criados hardlinks. A reutilização depende
de SHA integral e da auditoria de áudio/tensor antes de aceitar o novo cache.

Estimativa de cache completo:528.816.194 bytes, aproximadamente0,529GB.
Com hardlinks dos950 objetos que passarem pela conferência, o incremento
estimado é43.166.794 bytes, mais32MiB conservadores para temporários. Sem
reutilização, o pico adicional é o cache completo mais essa margem; o cache
antigo continua presente até uma limpeza autorizada e verificada. O treino
posterior acrescenta até1.648.868.862 bytes para last, best distinto e escrita
atômica sobreposta, antes de qualquer retenção ASR adicional. Antes de cada
etapa é obrigatório atualizar o uso físico global40GB e a reserva5GiB;
os26,8GB livres observados no volume não substituem essa conta global.

O preparo histórico dos990 objetos levou137,185s com quatro workers,
508,308s somando os workers. Extrapolação com dois workers:5–8min para
recalcular tudo, ou1–3min com reaproveitamento verificado. Essas são
estimativas CPU, sujeitas à concorrência do treino/ASR; não medem latência
do produto. Não foi feita nova extração para confirmá-las.

## Pitch e inicialização

Os rótulos persistidos são WORLD/DIO+StoneMask em **Hz**, interpolados nos
centros do mel. O collator normaliza apenas valores não zero:
`(Hz - mean_treino) / std_treino`; zeros permanecem zero. `prepare` calcula
mean/std exclusivamente de todos os vozeados do manifesto chamado `train`.
Reutilizar features brutas de outro split é possível; reutilizar as suas
estatísticas não é. As antigas186,86485/52,16814Hz vêm das810 falas e ainda
não são as estatísticas das1.025. As públicas são137,83144/41,93769Hz.

Após preparar1.025/55, conferir estatísticas finitas, contagem positiva,
distribuição de F0 e erros de oitava/vozeamento nos alvos robóticos. Essa
medição é indício auxiliar. A receita carrega o modelo público estritamente
e **reaplica os buffers pitch_mean/std da etapa nova** depois do load. O
collator, os buffers, o config salvo e qualquer transformação explícita de
pitch devem usar a mesma convenção. `infer` comum condiciona em pitch
normalizado e não converte automaticamente a saída para Hz; os buffers são
consultados quando há `pitch_transform`. A mudança de coordenadas exige
novo gate de pronúncia e identidade.

A inicialização será `--init-checkpoint` de um checkpoint público **qualificado
e recuperável por áudio**, com SHA integral e origem congelada. Não presumir
que mel-best5.500, last ou o passo6.000 sem pesos seja essa escolha. Começar
em diretório Aegis novo sem `--resume`: só modelo strict; Adam, scheduler,
scaler, RNG, passo, época e cursor novos. O piloto de memória usa o maior
alvo inteiro18,633s e um lote representativo do pico; depois o treino
científico começa novamente do público selecionado, não do piloto.

## Receita futura, ainda não executada

Reproduzir somente a preparação leve:

```sh
cd /home/phc/aegis-v2-lab
/usr/bin/python3 -m training.v2_fastpitch_target_plan
```

Depois da liberação do root, da conferência do orçamento e dos950 SHA
integrais, hardlinkar somente os objetos listados em `reuse-plan.json` para
`fastpitch/aegis-features-v2/features/<id>.pt`. Não copiar o diretório antigo
nem os seus manifests/stats. Se algum objeto não passar, extraí-lo novamente
com o mesmo helper. O payload de extração, supervisionado com dois workers
de uma thread e pausa das janelas de latência, será:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 nice -n 19 \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  training/fastpitch_experiment.py prepare \
  --manifests /var/tmp/aegis-v2-work/fastpitch/aegis-preparation/train.jsonl \
  /var/tmp/aegis-v2-work/fastpitch/aegis-preparation/validation.jsonl \
  --output /var/tmp/aegis-v2-work/fastpitch/aegis-features-v2 \
  --workers 2 --pitch-method world
```

Isso também recarrega e verifica os objetos reaproveitados e recalcula as
estatísticas só com as1.025 falas. Executar depois `audit_features` do helper
fixado sobre os1.080 objetos: SHA integral de cada feature e áudio, texto,
tokens, formato, shapes e finitude. Essa auditoria ainda não aconteceu.

Quando o público qualificar, pela fila GPU exclusiva, o comando abaixo é a
base proposta para o novo estágio; manter inicialmente os parâmetros
científicos públicos e confirmar recursos no piloto antes de iniciar:

```sh
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  training/fastpitch_experiment.py train \
  --features /var/tmp/aegis-v2-work/fastpitch/aegis-features-v2 \
  --output /var/tmp/aegis-v2-work/fastpitch/aegis-specialization-v2 \
  --init-checkpoint "$AEGIS_FP_PUBLIC_CP" \
  --device cuda --precision bf16 --batch-size 8 --accumulation 2 \
  --lr 0.0002 --warmup 1000 --max-steps 200000 \
  --validate-every 500 --checkpoint-seconds 1800 \
  --binary-start 1000 --binary-warmup 5000 --patience 12 --min-delta 0.01 \
  --max-lr-reductions 1 --plateau-lr-factor 0.3
```

`AEGIS_FP_PUBLIC_CP` só será definido após seleção do checkpoint público.
O limite de passos é uma fronteira ampliável; a parada depende da validação
e do áudio. Retomadas posteriores deste mesmo estágio usam `--resume` e a
mesma inicialização congelada; não restaurar o Adam do estágio público.


## Execução CPU autorizada e concluída

Em 03/11/2026, 14:50:20–14:51:29 UTC, o supervisor CPU separado
`training/v2_fastpitch_target_features.py` executou a etapa com CUDA oculta,
nice 19 e dois workers de uma thread. SHA256 da fonte:
`1c88782a8e3b52382d6f9dce00bf36b707ebf83f6ddc1cc1ea38beb9753876bc`.
PID 229461/start 372353168 supervisionou o grupo próprio
PID 229462/start 372353177; ambos encerraram com exit 0. O mecanismo STOP/CONT
do próprio grupo verificou UID, PID/start, PGID e SID; um teste stdlib
confirmou a pausa/retomada e a recusa de start ticks incorretos. A pausa não
foi necessária durante esta execução. Não houve GPU, fila ou sinal a trainer.

Os 950 objetos legados passaram integralmente por SHA de feature e áudio,
proveniência, texto/tokens, config, shapes e finitude antes dos hardlinks.
Foram preparados 130 objetos novos. O cache antigo e os originais continuam
intocados. O novo cache tem 1.025 treino/55 validação em
`/var/tmp/aegis-v2-work/fastpitch/aegis-features-v2/`, com 527.216.640 bytes
alocados. A auditoria final dos 1.080 objetos passou e hashou
1.980.553.840 bytes de áudio em 12,42 s; o worker completou em 63,92 s.
O log limitado ficou com 111.397 bytes.

Os stats recalculados só com 1.025 treino são mean 186,49572123592918 Hz e
std 52,02665106362999 Hz: 1.005.399 frames positivos entre 1.221.052. O intervalo
positivo 30,804–773,583 Hz merece inspeção futura de vozeamento/oitavas; não
comprova timbre. A checagem independente das somas reproduziu os valores
salvos em features.json. O helper 453c não foi alterado.

O orçamento global final foi 37.625.905.152 bytes, com 26.659.119.104 bytes
livres. O pico conservador inicial incluía 530 MB de features, 32 MiB de
temporários e 55 MB para checkpoint LoRA e ficou dentro de 40 GB/5 GiB.
A reutilização elevou o uso global em aproximadamente 40,44 MB. Não se fez
limpeza de insumos antigos nesta etapa.

Recibo principal: `aegis-features-v2/qualification.json`,
status `qualified_cache_only_not_model`. Manifests de features:
treino `b0eb4132920c47be06a17e2efd1494498059103dbc3cf46ad8f6510359eb6d72`,
validação `d09c52b17c8d8f7d02156abc47f5f8263c6c0c5d1efe89d3937cfce15517afc3`;
config `d117d7f0bdaec1551734aab24abb0801a06e157628497934a3fbef75cd6aa40c`.
Logs, orçamento, auditoria legada, teste de pausa e progresso estão em
`fastpitch/aegis-preparation/`. O teste final permaneceu fechado. Este cache
não autoriza treino Aegis antes de qualificar o idioma do checkpoint público.
