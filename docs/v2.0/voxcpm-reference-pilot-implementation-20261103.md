# Piloto independente0586: pronto para revisão, sem execução GPU

`training/v2_voxcpm_reference_probe.py` prepara por padrão um plano CPU. Seu
SHA-256 é `0fcb7652f834dbad8ac30368593d4f36673966ad6e6939ab2187c2d698556776`.
Nenhum trainer, runtime, helper compartilhado, perfil ou serviço foi alterado.
A hipótese e a auditoria da referência estão em
[voxcpm-train-reference-control-20261103.md](voxcpm-train-reference-control-20261103.md).

O piloto começa no checkpoint800 preservado (`b9160041…501f33`). Usa os três
casos de validação command/pressure/long, frontend `ptbr-numbers-v1`, NFE10,
CFG2, seed40733133, max_len400, LoRA r8 BF16 sem fusão e dois KVs1024.
Não lê o teste final nem usa âncoras de validação como condicionamento.

Antes da matriz, compara nos três casos os hashes FLOAT/PCM, comprimento e
duração entre `generate_streaming` sem referência e a API cacheada com None.
Uma diferença interrompe a execução e conserva os dois áudios para diagnóstico.
Ambos os modos A/B da matriz usam `generate_with_prompt_cache_streaming`;
B recebe o cache acústico fixo TRAIN0586, sem legenda/continuação/merge.

São6 chamadas de equivalência,18 warmups excluídos (3/caso/modo), depois18
chamadas medidas (n3/caso/modo, ordem A/B alternada). O cache é construído
uma vez: mede custo frio do VAE, pico de VRAM, forma real `[64,4,64]`, dtype,
hash e residência dos pesos/VAE/KVs na GPU. As características de entrada
retornadas pela API ficam na CPU; staging e prefill por pedido entram no tempo.

O guard da entrada cacheada cobre None/reference-only: L=T+1 ou R+T+3;
reserva L+400+1≤1024. Recusa overflow, formato/dtype/device inesperado e
qualquer modo de continuação. O hook compartilhado de capacidade é usado
apenas no carregamento e restaurado antes da medição. O gerador original é
fechado em EOF, erro e cancelamento; hooks temporários e handlers restauram
em finally. O FIR/coletor/métricas são os helpers existentes.

A saída contém JSON por chamada e agregado, WAV FLOAT48k/PCM24k de rep0 e
falhas, traces dos chunks, hashes e métricas TTFA/EOF/RTF/gaps/onset comuns.
Capturar FLOAT copia cada chunk dentro do tempo nos dois modos. Arquivos e
traços são gravados depois do EOF medido. Término da medição não aprova voz.

Os guards exigem worker proprietário como pai, GPU desocupada, pause-asr e
ASR inativo; cache≤512MiB, saída≤128MiB, orçamento40GB e5GiB físicos livres.
A reserva extra de1GiB inclui128MiB para sobreposição de checkpoint. Há
watchdog de orçamento herdado, sem criar outro gerenciador de cache. Os
hashes de código/fonte/configuração/checkpoint/referência são conferidos
antes/depois; pesos-base/VAE são verificados no início e após medição normal.
Falha interrompida não declara que essa última verificação dos pesos ocorreu.

Testes CPU:7 novos mais15 existentes de FIR/cache, **22/22 passaram**, sem
skips. Cobre fronteiras556/557tokens com referência,622/623 sem referência,
proibição de continuação, despacho, restauração, fechamento upstream em
EOF/cancel/erro e cronograma A/B. Torch2.6.0+cu118, CUDA não inicializada,
pico400,29MiB; nenhum modelo TTS foi construído. Usei unittest do ambiente
existente; pytest não está instalado nele e nenhuma dependência foi adicionada.

Artefatos privados:

- `/var/tmp/aegis-v2-work/voxcpm2/reference0586-step800/plan.json`:
  SHA `b99bbf3f76a61f5a757c869c1fa388dd958b17365855194a95abe2e1fc4448e1`.
- `/var/tmp/aegis-v2-work/voxcpm2/reference0586-control/cpu-pilot-tests.json`:
  SHA `7b5d9a6b9cacf6aca6d66fea904046835e27643cff94392059232f68fc8b14a0`.

Comando candidato, **somente para agendamento posterior no worker**:

```bash
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_voxcpm_reference_probe --action execute \
  --run /var/tmp/aegis-v2-work/voxcpm2/reference0586-step800 \
  --cache /var/tmp/aegis-v2-work/voxcpm2/compile-cache \
  --checkpoint /var/tmp/aegis-v2-work/identity-aux/snapshots/voxcpm2-step800-78680488bcf4/checkpoint.pt \
  --checkpoint-sha256 b9160041e0773799ab20e2066fb526f7151425bb8055491add375b1007501f33 \
  --gpu-authorized
```

Apenas `--action plan` e os testes CPU foram executados. Não houve CUDA,
fila, interrupção do treino, síntese neural ou alegação de melhora. O resultado
nativo também não representa o gateway/startup320ms: essa confirmação continua
dependente de uma referência que justifique avançar e de nova medição real.
