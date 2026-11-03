# Revisão independente da integração Vox no gateway

Revisão somente leitura da integração experimental, sem modelo ou GPU.
Fonte de runtime revisada: SHA256
`52299d3b47238771ef3135e8d0ac436edbe1d86cd28bdc66e110c1e0909ac2f2`.
Não identifiquei impedimento restante para os testes de contrato com backend
fake. A carga real do modelo e os gates do produto continuam pendentes.

Três achados foram enviados ao autor e corrigidos: texto original com
números/endereço técnico podia conservar placeholders do segmentador antes
de uma sentença posterior; falha no reset podia impedir liberação do
modelo/lease; inicialização cancelada podia pular session.close. As correções
ficaram no frontend experimental e nos finally de encerramento. O caminho
padrão Qwen e o segmentador v1 continuam preservados.

Provas independentes stdlib confirmaram o texto completo de3.14/IP seguido
por outra sentença, liberação após exceção de reset e fechamento de
sessão/executor com inicialização cancelada. Não importaram Torch. A tentativa
de executar pytest no ambiente F5 encontrou pytest ausente; nenhuma
dependência foi instalada. A suíte fake do autor usa o ambiente de testes
já disponível e fornece seu próprio recibo.

A inspeção cobriu carga de base/VAE/LoRA estrita e pinada, inferência BF16
sem merge, residência GPU, thread proprietária única, três aquecimentos,
backpressure e cancelamento, FIR com estado, fim por limite e integração
PCM/WAV. O upstream pinado emite um chunk por patch; o guard de fim é
conservador quando a contagem atinge o limite, sem alegar sucesso completo.

Isso não prova CUDAgraphs reais no processo gateway, latência, saúde do
modelo carregado ou identidade. Antes de disponibilizar a candidata,
recarregar os pesos selecionados em processo novo e validar prontidão,
WAV completo, PCM em streaming, cancelamento e a matriz comum do gateway.

Recibo privado com hashes de todos os arquivos e limites da revisão:
`/var/tmp/aegis-v2-work/voxcpm2/aegis-lora-preparation/gateway-independent-review.json`.
Nenhum arquivo da integração, serviço, fila ou peso foi alterado pelo revisor.
