# Revisão independente do runner LoRA VoxCPM2

Fonte revisada: `training/v2_voxcpm_train.py`, SHA256
`fb45b037e3e2fe29eacfa359b2196a5079f6f8629fb11278ff98d2d1018f70e6`.
Revisão somente leitura do runner, testes e receita; nenhum peso grande foi
carregado e nenhuma tarefa GPU foi enfileirada. Não encontrei impedimento
para o piloto exclusivo com a fala inteira de 18,633 s. Isso não aprova o
modelo treinado nem substitui a prova GPU de três updates e retomada em
processo novo para o quarto.

A chamada direta ao packing oficial preserva texto, máscara de perda só no
áudio e stop no último patch. O wrapper rejeita sequência excessiva sem
cortar a fala. A perda soma difusão e stop nativos; autocast BF16 conserva a
rota de ativações enquanto LoRA e Adam têm estados FP32. O loader congela
VAE e backbone, e o controlador devolve o VAE a eval após validação.

Os cinco testes CPU foram reexecutados no ambiente F5: passaram em 5,062 s,
sem inicializar CUDA. Incluem 3→checkpoint→modelo novo→4 versus quatro
updates contínuos, estados Adam/dados/RNG, cache de inferência antes de
backward, reconciliação da publicação interrompida e rejeição de Adam parcial.

Uma prova independente também chamou o `synthesize` real com LoRA FP32 de
valores não representáveis em BF16. Tanto no caminho normal quanto em uma
exceção deliberada, os masters e ponteiros originais permaneceram idênticos,
os momentos Adam ficaram intactos, o RNG foi restaurado e o backward
posterior produziu gradientes finitos. Somente a telemetria CUDA foi mockada;
não se trata de inferência do modelo real.

O snapshot exige cobertura integral dos parâmetros Adam após o primeiro
update, momentos com formas/dtype corretos, passo consistente, scheduler,
scaler BF16 desativado, RNG, permutação e cursor. Best/last usam temporário
validado, fsync, hash e intenção persistida para reconciliar queda entre peso
e sidecar. A configuração fixa dados/base, agenda e critério de parada;
alteração de código ou recurso exige override explícito registrado. O limite
de updates pode ser ampliado, sem prazo de parede usado como reprovação.

Permanecem abertos: memória e custos na fala longa, todos os 384 estados no
checkpoint do novo runner, retomada GPU real depois das sínteses, ASR dos
sete textos e identidade vocal. A perda de validação com áudio conhecido é
auxiliar; só áudio gerado de texto inédito pode sustentar essas decisões.

Recibos e probe reproduzível, privados e pequenos:
`/var/tmp/aegis-v2-work/voxcpm2/aegis-lora-preparation/independent-training-review.json`
e `independent-training-review-probe.{py,json}` no mesmo diretório.
