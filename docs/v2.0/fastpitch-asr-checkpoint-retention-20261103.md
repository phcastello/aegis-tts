# FastPitch: retenção por pacote nativo e ASR

Supervisor separado em `training/v2_fastpitch_checkpoint_retention.py`,
lançado pelo root em 03/11/2026 às 13:28:13 UTC, PID 221380,
start ticks 371860532, source SHA256
`d521a9ba195ebb199f0dc7bde784c4ae17d93300b799a02e40a99be2ff49aea0`.
Não altera
trainer, monitor ativo, dados ou `last.pt`/`best.pt`. Não carrega Torch nem
executa ASR/GPU ou envia sinais ao treino.

O passo 6.000 teve ASR nativo melhor, mas seu checkpoint já não estava
disponível quando a pontuação chegou. O supervisor não tenta reconstruí-lo
nem substituí-lo pelo mel-best 5.500 ou last 7.418. Esse caso fica marcado
como indisponível.

A conferência leve das 13:44 UTC vinculou PID, start ticks, UID e comando ao
recibo. O processo tinha uma thread, cerca de 19 MiB de RSS e nenhum
mapeamento Torch/CUDA. Após o fim da janela HTTP estava `paused=false`,
com zero snapshots e sem melhor recuperável. O passo 6.000 continuava
explicitamente indisponível; não houve captura do 7.418. Prova privada:
`/var/tmp/aegis-v2-work/fastpitch/language/asr-checkpoints/launch-verification.json`.

Ele lê o estado do quality-monitor, `validation-latest.json` e
`checkpoint.json`. No passo previsto pelo cadence do monitor, pode congelar
o pacote nativo antes de terminar o ASR. Só publica o checkpoint quando o
passo coincide e o SHA256 do arquivo completo coincide com o recibo do
trainer. O pacote congelado também tem SHA integral. O resultado ASR só é
associado se seu passo e SHA do pacote forem os mesmos. Uma publicação
posterior do trainer pode avançar o path; o inode já vinculado continua
válido pelo hash anterior. Publicação incompleta ou recibo de outro passo
fica para a próxima verificação, sem desserializar pesos.

Intenção persistida, fsync e rename permitem recuperar uma interrupção entre
hardlink e manifest, mesmo após o trainer avançar. O supervisor mantém no
máximo um estado pendente e o melhor ASR recuperável da política corrente.
Enquanto um pendente aguarda pontuação, não abre outro slot; um link antigo
ainda em uso também bloqueia captura nova. Se o monitor de ASR escolher
outro pacote sem pontuar o antecipado, esse último pode ser descartado.
O melhor observado sem pesos continua distinto do melhor recuperável.

Artefatos futuros ficam somente em
`/var/tmp/aegis-v2-work/fastpitch/language/asr-checkpoints/`:
checkpoints e pacotes por hardlink, manifests/sidecars, estado, avisos e
recibos de limpeza. Remoção se limita à própria árvore, conferindo UID,
path real e fuser. O registro distingue blocos dos links removidos de blocos
exclusivos efetivamente liberados. Os paths de retomada do trainer ficam
intocados.

`/var/tmp/aegis-v2-work/pause-asr` suspende todo trabalho pesado do
supervisor: a iteração retorna antes de descobrir/reconciliar snapshots,
calcular hashes, percorrer o orçamento, capturar ou limpar. Durante a pausa
somente o status leve (`paused=true`) e o heartbeat normal são publicados.
Uma inicialização durante a pausa também adia a recuperação de snapshots.
Antes de remover staging duplicado ou órfão, verifica `fuser` tanto no
diretório quanto nos arquivos; se houver uso, preserva o conjunto e registra
aviso para revisão.

Antes de capturar, mede o orçamento global por inodes únicos. Exige espaço
para conservar o checkpoint, mais pelo menos 550 MB de sobreposição com a
próxima gravação e 64 MiB para o pacote, dentro de 40 GB, além da reserva
física de 5 GiB. Falta de espaço gera aviso e bloqueia somente a captura.
Há lock único, polling local de 2 s, logs limitados a 1 MiB e heartbeat de
30 min. Não existe aprovação automática de idioma ou identidade.

Dez testes stdlib passaram: publicação tardia, troca do path durante SHA,
recuperação após hardlink antes do commit, vínculo pacote/checkpoint/ASR,
orçamento com sobreposição, limpeza preservando last/best originais,
retorno sem trabalho pesado sob pausa, inicialização pausada e preservação
de staging duplicado/órfão em uso.

Comando do supervisor lançado após revisão, em sessão desacoplada como os
demais supervisores:

```sh
cd /home/phc/aegis-v2-lab
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 \
  -m training.v2_fastpitch_checkpoint_retention \
  --run /var/tmp/aegis-v2-work/fastpitch/language \
  --root /var/tmp/aegis-v2-work
```

`--once` executa uma única inspeção/captura e não deve disputar o lock do
supervisor ativo. Os testes usam somente arquivos temporários pequenos.
