# Piper Cadu: implementação pronta para o piloto isolado

Atualização posterior:084 falhou antes do update1;085 retomou o inicial0 e avançou0→4. A prova nativa3→4 continua aberta. A correção de contexto/cache e o novo piloto estão em [piper-autograd-cachefix-20261103.md](piper-autograd-cachefix-20261103.md). Os números CPU abaixo pertencem à implementação inicial.

`training/v2_piper_experiment.py`, SHA256`71309a7198136d4fe00e9eff0e968cf46db00a59fa91df82eb40e1b18da787e2`, implementa `fetch`, `prepare`, `check`, `reconstruct`, `pilot`, `train` e `infer`. Não usa Lightning. Reutiliza fonte Piper1.8 fixada, os modelos/losses originais e a MAS CP312 existente com transposição conferida. Todos os pesos G+D são exigidos; uma nova especialização inicia Adam/schedulers/RNG/contadores novos. A própria retomada restaura dois Adam, dois schedulers, scaler desativado FP32, RNG Python/NumPy/TorchCPU/CUDA, época, permutação, próximo cursor, contador de pares completos e configuração/hashes. O checkpoint contém hparams e redes completos e permite retomar sem reler a base pública.

O checkpoint Cadu original foi baixado uma vez, em32,55s, com845.888.150bytes e SHA256`abd3e6c7237aca5b7cf9c3cfc50945bfef44a83a594e00186f584d9d11fd355a`. Houve hash durante download, releitura local, fsync e publicação atômica. Arquivo: `/var/tmp/aegis-v2-work/piper/downloads/pt_BR-cadu-medium.ckpt`; receipt:`piper/full-download-receipt.json`. Pesos são privados para pesquisa; a ressalva da linhagem Lessac permanece e nenhum upload foi feito.

## Evidência CPU real

* Carga estrita de673tensores G +111D, todos finitos;23.663.792/46.747.132parâmetros; Torch2.6.0+cu118; `cuda_initialized=false`. Receipt:`piper/cpu-strict-check/cpu-check.json`.
* Frontend nativo pré-computado em subprocesso:1.025treino/55validação, mapa161/nasalU+0303=141, zero OOV. Teste50 não foi lido. Manifesto privado:`piper/target-native161.json`, fingerprint`e9ae98ce21f2974a86e7a10f3fb5638d22f356a4c69611222205c4bc8a1d6a67`. Usa WAVs preparados24k verificados, conversão22.05k em RAM e LRU limitado; não expande áudiofloat/espectrogramas no disco.
* Síntese PyTorch CPU da frase inédita `v_command`:66.304samples22.05k,3,007s finais24k, pico0,5644, sem overshoot/clipping. Mesmo número de samples do ONNX0/0; diferença máxima3,02e-5, RMS1,35e-6. Artefatos:`piper/cpu-native-infer/`.
* Reconstrução de uma âncora reservada:11,726s, pico0,3633, sem overshoot/clipping. Par alvo/reconstrução em FLOAT22k e PCM24k, hashes e auxiliar em`piper/cpu-posterior-reconstruction/`. MAElog-espectral0,785; RMS−1,30dB; centroide−118,8Hz; duração−5,99ms. Não foi feita escuta nem avaliação de identidade. Essas métricas não elegem candidata.

Os diagnósticos de áudio expõem offsets internos de início de fala, sem atribuir chegada artificial0ms a TTFA. Não são benchmark de latência. A geração CPU usou o core de código`c810cb3a...`; a versão atual corrigiu apenas o formato QC/metadados, adicionou o alvo pareado e bloqueou uso ambíguo de `--checkpoint` em treino. Nenhuma fórmula de geração mudou.

Seis testes pequenos CPU passaram: continuação3pares→save→backendnovo→4 idêntica a4contínuos em pesos G+D, momentos/steps Adam, RNG e cursor/permutação; recuperação de falha entre peso e recibo, preservando best anterior; rejeição de Adam incompleto/step errado/permutação/NaN; preservação RNG; MAS retangular/guarda impossível; QC sem latência fictícia. Testes não carregam o checkpoint completo. Compilação/help/diffcheck passaram.

## Execução seguinte

Todos os caminhos abaixo são privados sob `/var/tmp/aegis-v2-work/`. Trabalhar em `/home/phc/aegis-v2-lab` com Python `/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python`. `check`, `infer --case v_command` e `reconstruct` já executaram em CPU. Comandos GPU preparados, **ainda não executados nem enfileirados por este agente**:

```text
python -m training.v2_piper_experiment pilot --device cuda --gpu-authorized --run /var/tmp/aegis-v2-work/piper/gpu-engineering-pilot --max-updates 3
python -m training.v2_piper_experiment pilot --device cuda --gpu-authorized --run /var/tmp/aegis-v2-work/piper/gpu-engineering-pilot --resume /var/tmp/aegis-v2-work/piper/gpu-engineering-pilot/last.pt --max-updates 4
python -m training.v2_piper_experiment infer --device cuda --gpu-authorized --checkpoint /var/tmp/aegis-v2-work/piper/gpu-engineering-pilot/last.pt --case v_question --run /var/tmp/aegis-v2-work/piper/gpu-fresh-infer
```

Passar pela fila exclusiva após liberar a GPU e medir o orçamento. A linha não promove modelo automaticamente. O melhor começa na validação da base e possui checkpoint próprio recuperável; regressão não substitui best. Só há last/best, hardlink quando coincidem; cada gravação valida o temporário em CPU antes do rename, com recibo pendente reconciliável após queda. Atualizações só são publicadas em fronteira G+D completa. Logs/progresso são limitados; checkpoint≤1.800s/época; sondas7nativas0/0 sem teacher forcing preservam RNG e retêm seis snapshots recentes mais o best/inicial.

Pico previsto do piloto inclui dois estados Adam completos e o best inicial sem momentos (~282MB), além da base pública já presente, cacheRAM128MiB e reserva do FastPitch. Treino longo exige folga para best/last/temporário completos; o programa mede uso físico global e preserva5GiB livres no volume. O supervisor principal fará eventual compactação/limpeza Coqui; este agente não apagou esses artefatos nem pesos Vox.

Ainda faltam update real Piper/A2000, retomada nativa3→4, VRAM/custo por passo e comparação de idioma/identidade após adaptação. O gate de idioma da base permanece provisório; esta implementação não converte o ASR da base em aprovação da voz-alvo. Plano detalhado:`NEXT-piper-gpu-pilot.json`.
