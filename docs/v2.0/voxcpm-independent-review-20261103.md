# Revisão independente do piloto VoxCPM2 — 03/11/2026

Revisão de `training/v2_voxcpm_probe.py` e da fonte limpa no commit
`cce58a5b59303c9bd63d12f23afe6a49a4a80c59`, sem carregar novamente os
pesos grandes e sem executar GPU. O piloto está apto à próxima prova de
engenharia local, depois da execução VITS; isso não aprova PT-BR, identidade
ou desempenho na A2000.

## Packing e losses

A ordem passada ao `forward` corresponde à assinatura: text tokens/mask,
audio features/mask, loss mask, position IDs, labels. O packer upstream é
usado diretamente, evitando truncamento de `__call__`; o limite512 rejeita
sequência longa integral, sem recortar áudio ou transcrição.

O packer efetivo foi executado em CPU com um VAE substituto determinístico,
hop640/16kHz, patch4 e dimensão64. Inputs9600,9601 e96000samples produziram
4,4 e38patches. Foram conferidos: tokens originais+101 antes do áudio,
102 no final, features zero nos lugares de texto, máscaras complementares,
loss apenas nos lugares de áudio e um único label stop no último patch.
O arredondamento é `ceil(samples/2560)`, como na fonte fixada.

No modelo, os hiddens e o condicionamento de features são deslocados uma
posição; isso combina com o label stop do último patch e a previsão causal
do próximo patch. O gerador acrescenta esse patch antes de testar o stop.
Diffusion e stop são losses upstream, com MSE/CE sob a política FP32 do
autocast; não foi executado um forward acústico completo nesta revisão.

## Precisão e checkpointing

O backbone congelado é construído em BF16 e os LoRAs trainable são promovidos
a FP32. A fonte converte o AudioVAE a FP32 **antes** de carregar seus pesos;
o encode do packer ocorre fora do autocast, com VAE congelado/eval. O loader
CPU mantém o dtype configurado BF16; o fallback a FP32 da função de escolha
de dtype aplica-se a MPS, não à CPU desse host.

A closure por layer captura método e módulo nos argumentos default, evitando
late binding. Prova CPU do wrapper real: duas classes pequenas com o mesmo
nome `MiniCPMDecoderLayer`, pesos2 e3 e input sem gradiente produziram6 e
gradientes distintos3/2. Isso verifica o caminho não reentrante com inputs
congelados; o RNG da recomputação é preservado. `forward_step` de inferência
permanece sem checkpointing.

Recibo dessas provas pequenas:
`/var/tmp/aegis-v2-work/voxcpm2/independent-packing-review.json`.

## Estados, inferência e áudio

O optimizer CPU opcional conserva masters FP32 separados, copia gradientes
após clipping e atualiza de volta os LoRAs que permanecem na GPU. Retomada
restaura adapters, masters, optimizer, scheduler, RNG e cursor. A ordem dos
clipes é determinística, e configuração/manifest/hash do código são vinculados
ao checkpoint. O target step deve avançar; o piloto normal é3→4 em processos
distintos. O número de passos é limitado a10; não é receita de treino longo.

A revisão identificou a necessidade de rejeitar adapters incompletos, que
o autor corrigiu com set, shapes, dtype e finiteness exatos em restore/infer.
O autor também verifica pesos/buffers, caches KV não registrados e atributos
derivados de weight norm antes de declarar residência GPU. O check CPU real
do autor confirmou keys base/VAE,384tensores LoRA e4.521.984parâmetros FP32;
o recibo `voxcpm2/cpu-check/cpu-check.json` é separado desta revisão. Somente
um piloto GPU poderá comprovar o consumo e a residência efetivos na A2000.

O caminho acústico usa16kHz no encoder e48kHz no decoder, com resampling
Kaiser10 explícito. O roundtrip remove apenas o padding final além do
comprimento de entrada reamostrado×3; não deve ser chamado de preservação
de banda de48kHz. Overshoot é conservado em FLOAT e contado, e conversão
PCM de produto é rejeitada quando ultrapassa a faixa. O piloto de inferência
mede arquivo completo, incluindo resampling24kHz e fechamento WAV; não
atribui streaming hipotético nem latência HTTP.

Para eventual etapa pública, usar `text` revisado em vez de ignorar revisões
por `original_text`; o manifesto alvo atual não depende desse ajuste. Também
convém rejeitar roundtrips vazios/não finitos e registrar comprimentos antes
do corte de padding. Nenhuma conclusão perceptiva foi inferida dessas provas.
