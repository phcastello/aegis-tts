# Revisão independente: cache guard e frontend numérico

Delta revisado somente leitura, runtime SHA256
`76b9ea179e290a2fa16f6e85947d021042a407cecb551a6f7f87bf12caa6395c`.
Nenhum impedimento restante identificado para repetir o benchmark HTTP
exclusivo. O ganho de latência HTTP e o áudio desta versão continuam abertos;
os números de custo CPU do autor não foram apresentados como TTFA.

O runtime passou a manter uma única thread de baixa prioridade para varrer
cache/volume. O owner CUDA lê snapshots em generate/observe/reset e verifica
o estado entre chunks e durante espera de backpressure. A inicialização
espera uma leitura inicial e outra depois dos três aquecimentos. Exceção,
thread morta, snapshot vencido ou fechamento impedem prontidão/continuação.
O período é5 s e a idade máxima15 s; a margem de1 GiB do executor e5 GiB
livres continuam necessárias porque limites podem ser cruzados entre leituras.

Foi encontrado e corrigido um caso de concorrência: a primeira versão do
force-fresh aceitava a conclusão de uma varredura iniciada antes do pedido.
Agora o início da leitura também precisa ser posterior ao pedido. Uma prova
independente stdlib bloqueou uma leitura já em curso e confirmou que o
chamador continuou esperando a próxima. Também confirmou despertar com erro
ao fechar, join de todas as threads e ausência de import Torch/modelo/GPU.

O frontend `ptbr-numbers-v1` usa as regras numéricas existentes com lexicon
vazio: preserva Aegis/CPU/API/APIs e expande14h05 e2,7 kHz. As variantes têm
perfil/header/texto ASR efetivo explícitos; original NFC permanece padrão e
F5 completo continua separado. O caminho Qwen protegido não foi alterado.
Não houve avaliação auditiva nem aprovação de pronúncia nesta revisão.

Recibos e prova reproduzível privados em
`/var/tmp/aegis-v2-work/voxcpm-gateway/preparation/`:
`independent-cacheguard-review.json`,
`independent-cacheguard-probe.py` e `independent-cacheguard-probe.json`.
O revisor não alterou fontes compartilhadas, fila, serviços ou pesos.
