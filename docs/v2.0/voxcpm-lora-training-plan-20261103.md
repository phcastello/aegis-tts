# VoxCPM2: especialização Aegis LoRA preparada

Hipótese: LoRA nos LMs e no DiT aprende o timbre robótico dos alvos sem perder
o PT-BR já qualificado. Avançar depende de áudio inédito, ASR e identidade;
perda de difusão/stop menor não promove a voz. O treino completo só deve
começar depois do gate PT-BR comum e do novo piloto de memória/retomada.

Implementação: `training/v2_voxcpm_train.py`, sem alteração dos helpers ou da
fonte upstream. Reutiliza o loader BF16, AudioVAE FP32, LoRA FP32/rank8 com
LM+DiT/sem projeções e checkpointing não reentrante dos blocos do piloto.
Treino eager, microbatch1/acumulação4, AdamW CUDA, LR1e-4/warmup100. A opção
`--saved-tensors-cpu` transfere somente tensores salvos do autograd para RAM;
a inferência continua integralmente na GPU. Seu custo/memória ainda exige o
piloto local. Nenhum treino deste novo runner foi executado em GPU nesta
subtask.

Os manifests privados congelados têm 1.025 falas de treino/13.030,096 s e
55 de validação/722,994 s. O runner não carrega o teste de 50 alvos nem as seis
frases finais. Usa `original_text` em NFC, sem reescrever números ou aplicar
frontend extra. Lê os WAVs originais vinculados por SHA256, reamostra para16k
com Kaiser10 e conserva a fala inteira, sem corte, ganho, denoise ou trimming.
O cache derivado é somente RAM, LRU256MiB. O maior alvo de treino é
`aegis_ptbr_0807`,18,633s; o maior de validação tem16,814s.

O packing chama diretamente `process_tts_data`, preservando máscaras,
características e rótulo stop do piloto auditado. Não usa o batching que pode
limitar o comprimento. Sequência acima do limite explícito falha em vez de
recortar áudio/texto. A perda é somente `loss/diff + loss/stop` nativa.

A validação usa todos os55 alvos com ruído fixo por índice e sete sínteses
inéditas comuns, sem DSP extra. O cache de inferência é1024, com a mesma
guarda explícita de entrada e parâmetros/RoPE originais. Todo RNG de treino
é restaurado. Tensores persistentes criados em `inference_mode` são clonados
fora desse modo antes do próximo backward. As sínteses são eager para
inspecionar conteúdo; a comparação final exige novo processo, pesos salvos,
LoRA real e streaming compilado, sem supor equivalência de merge BF16.

Cadência padrão: validação200updates, checkpoint no máximo30min, paciência8,
uma reduçãoLR0,3 e nova paciência. `max-steps` é fronteira intermediária
ampliável, sem prazo de parede. Logs próprios retêm5MiB; validações compactas
ficam preservadas. Áudios retêm últimas3 avaliações, inicial e atual melhor.

`last.pt`/`best.pt` contêm LoRA, Adam completo, scheduler, scaler desativado
BF16, RNG Python/NumPy/Torch/CUDA, permutação, época/cursor, passo, seleção,
dados/configuração e hashes de código/base. O writer salva temporário,
fsync, recarrega em CPU, confere finitude, momentos/parametrização/passo,
permutação e RNG, compara LoRA exatamente e publica por rename com intenção
persistida e sidecar SHA256. Queda entre peso e sidecar é reconciliável.
Best pode compartilhar inode com last. Orçamento inclui512MiB de margem
para sobreposição/saídas, além da reserva física5GiB. Checkpoint inferível
mantém `lora` e `config.rank`, aceitos pelo loader nativo existente.

Cinco testes CPU passaram no ambiente F5:3updates e retomada em novo modelo
para4 equivalem bit a bit a4 contínuos, inclusive Adam/dados/RNG; cache de
validação não impede backward; save não altera RNG; crash de publicação e
Adam parcial são tratados. O checkpoint do piloto GPU anterior foi lido
apenas em CPU:384 tensores LoRA,384 estados Adam completos, todos no passo4.
Isso não substitui o novo piloto com os alvos completos18,63s.

Plano privado:
`/var/tmp/aegis-v2-work/voxcpm2/aegis-lora-preparation/plan.json`.
Fingerprint inicial dos manifests/seleção:
`3fe40f87b8dd5874c098bb404d6a1998826a220d67312b944c9e1a04adac6a68`.

## Comandos futuros pela fila GPU exclusiva

Piloto com a fala inteira mais longa, validação reduzida explicitamente
marcada como engenharia e síntese após cada update:

```sh
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_voxcpm_train --action train --gpu-authorized \
  --run /var/tmp/aegis-v2-work/voxcpm2/aegis-longest-pilot \
  --pilot-longest --validation-limit 2 --probe-limit 1 \
  --validate-every 1 --max-steps 3 --nfe 10
```

Depois de conferir pesos, sidecar e áudio, repetir o mesmo comando em outro
processo com `--max-steps 4 --resume`. Se faltar VRAM, preservar o último
estado e testar `--saved-tensors-cpu --resume-allow-resource-change --resume`,
registrando custo e RAM. Não existe fallback de inferência em CPU.

Após passar pelos gates, iniciar uma especialização nova; não usar o piloto
de uma única fala como inicialização científica:

```sh
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_voxcpm_train --action train --gpu-authorized \
  --run /var/tmp/aegis-v2-work/voxcpm2/aegis-lora \
  --max-steps 20000 --nfe 10
```

Retomar com os mesmos argumentos e `--resume`, ampliando `max-steps` se
necessário. Alterações de código/runtime exigem override explícito e recibo;
dados/base/receita científica diferentes são recusados. Parada por sinal é
concluída na fronteira de update; falhas parciais preservam o checkpoint
anterior. Nunca iniciar este runner junto de outro backend na GPU.

## Benchmark compilado do adapter real

`training/v2_voxcpm_lora_benchmark.py` é um wrapper separado, com NFE10 por
padrão para corresponder ao baseline PT-BR qualificado. Reutiliza o runner
nativo, `capacity_hooks(1024)`, guardas da fila GPU, pausa de ASR, teto de
cache de compilação512MiB e três aquecimentos por frase. Mantém os ramos
LoRA, sem merge. O loader copia os masters FP32 do checkpoint e transforma
somente a cópia de inferência em BF16; o wrapper verifica branches,
precisão/congelamento e residência CUDA reais. O SHA256 do arquivo e seu
sidecar são conferidos antes e depois. A receita adicional registra hashes
dos scripts, modelo/fonte, runtime e capacidades reais; o checkpoint de
origem permanece intacto. Os sete áudios posteriores ao treino continuam
necessários para ASR/identidade.

Depois do piloto3→4, executar pela mesma fila exclusiva:

```sh
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_voxcpm_lora_benchmark --action execute --gpu-authorized \
  --checkpoint /var/tmp/aegis-v2-work/voxcpm2/aegis-longest-pilot/last.pt \
  --run /var/tmp/aegis-v2-work/voxcpm2/aegis-pilot-lora-native \
  --nfe 10 --cases v_reply v_long --repetitions 3
```


Atualização de execução 12:43 UTC: o piloto GPU com o maior áudio inteiro
passou três updates e retomada3→4 em processo novo; pico6,332GB. O benchmark
compilado com LoRA não mesclada também passou os gates curto/longo, sem
lacunas. O run científico completo começou12:33:55 com inicialização nova
da base e já apresentou atualizações finitas e delta de pesos positivo.
Consulte o registro persistente privado e o journal para estado atual; o
teste final continua fechado e nenhuma identidade vocal foi aprovada.
