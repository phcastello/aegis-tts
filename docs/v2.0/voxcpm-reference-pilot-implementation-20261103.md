# Piloto0586 v2: equivalência após aquecimento fixo

A primeira execução foi feita pelo worker do proprietário em03/11,17:01–17:04.
O guard interrompeu antes da matriz de referência: primeira chamada legacy
fria/compilando e cachedNone tiveram o mesmo comprimento3,04s, mas hashes
diferentes. Diferença FLOAT máxima0,0002311/RMSE0,00001463; PCM máximo7LSB,
RMS0,492LSB, concentrados em0,115–0,509s. Isso não comprova causa nem permite
afrouxar a equivalência. Nenhum pedido condicionado B foi executado.

O cache da0586 foi construído corretamente: `[64,4,64]`, FP32 CPU65.536bytes,
3,393s frio, pico5629,5MiB e pesos/VAE/KVs integralmente na GPU. Esses dados
descrevem engenharia; não há comparação de conteúdo ou identidade A/B.
O treino retomou automaticamente do2000 e avançou, com delta de pesos real.

A auditoria final também encontrou um bug do runner: locale ASCII após
compilação fez a leitura sem encoding falhar. O repr de UnicodeDecodeError
incluiu o buffer privado inteiro em um recibo local. Esse recibo original
permanece privado e não deve ser publicado. A nova fonte usa UTF-8 explícito
em toda E/S textual própria e mensagem de exceção limitada a512caracteres,
sem repr. Uma conferência independente após o abort verificou todos os
hashes correntes contra a receita anterior; isso não substitui a auditoria
que falhou no instante original.

O controle autorizado agora executa **três rodadas fixas por API/caso**, com
ordem legacy/cached alternada:18 chamadas excluídas das métricas A/B. Compara
somente o último par, ainda estritamente FLOAT/PCM/comprimento/outcome, e
exige sucesso nas18 chamadas. Guarda todos os JSONs; pares diferentes e
falhas mantêm WAVs. Não repete até coincidir nem aceita tolerância. Só após
esse gate começam os18 warmups A/B e18 pedidos medidos anteriores.

Fonte v2: `adff50ab6e8d939075968eb326033ed8edd8bcc9de32e976c079fea15e87dc8d`.
**26 testes CPU passaram**, CUDAfalse, pico400,19MiB: incluem primeiro par
diferente/último estável, diferenças persistentes sem chamadas adicionais,
falha inicial que não pode ser ocultada e processo real sob locale C.
O plano completo também passou com `PYTHONUTF8=0 LC_ALL=C` nos dados reais.
Nenhuma execução GPU v2 foi feita por este agente; não foi criada fila.

Artefatos sob `/var/tmp/aegis-v2-work/voxcpm2/`:

- `reference0586-step800-warmed-api-v2/plan.json`,
  SHA `8e49b4adf937652bf8e1db697d24b111f3ab679d4281fb4dbad131a015ed0cf5`.
- `reference0586-control/cpu-pilot-tests-v2.json`,
  SHA `c42dc14f62fcc38726e4d5c3f75acf79ef392da025e64cd2636453c652d616bc`.
- `reference0586-control/failed-equivalence-cpu-diagnostic.json`,
  SHA `0b01e982f9f2e84b2e823a946e588d1b87d7a44f903b555524d5aa473e3e0198`.
- `reference0586-control/failed-first-cold-0fcb/`: fonte antiga copiada,
  quatro WAVs e recibos preservados por hardlink, originais mantidos.

Comando v2 candidato, apenas pelo worker em janela exclusiva:

```bash
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_voxcpm_reference_probe --action execute \
  --run /var/tmp/aegis-v2-work/voxcpm2/reference0586-step800-warmed-api-v2 \
  --gpu-authorized
```

Os defaults continuam vinculados ao mesmo CP800, referência0586 e cache512MiB.

## Preparação inicial preservada como histórico

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
