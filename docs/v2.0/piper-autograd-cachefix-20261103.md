# Piper: falha084, correção dos caches e novo piloto

O job084, fonte`71309a7198136d4fe00e9eff0e968cf46db00a59fa91df82eb40e1b18da787e2`, falhou **antes da primeira atualização**. A validação inicial havia criado caches globais Hann/mel dentro de `torch.inference_mode()`. No primeiro backward, `mel_spectrogram_torch`/`torch.stft` tentou salvar esses tensores para calcular gradientes e produziu `Inference tensors cannot be saved for backward`.

O job085 carregou o checkpoint **inicial step0**, epoch0/cursor0, e completou0→4. Houve mudanças reais em G/D; step4 levou1,1481s e pico Torch2.070,79MiB. O job086 recarregou e sintetizou. Esses resultados comprovam updates e recarga, mas **não comprovam o piloto exigido3→processo novo→4**. O melhor continuou sendo a base inicial; não há aprovação de identidade ou idioma da adaptação.

| Checkpoint histórico | Passo | Bytes | SHA256 do recibo validado |
| --- | ---: | ---: | --- |
| best inicial | 0 | 281.976.270 | `970479dc1c983e036979838a37219980e599e81434db6773d158c62dd9898de6` |
| last após085 | 4 | 845.928.392 | `0fff4dd0e80029460118d56b3a4065021ff0a6c2ed08504b1513a0982a78388d` |

Logs/recibos084–086 e `piper/gpu-pilot-084-085-reconciliation.json` registram a distinção. Este agente não removeu checkpoints; ao tentar nova releitura após comunicar a correção, os arquivos já não estavam no diretório antigo. Os hashes acima vêm dos recibos da publicação/carga histórica, sem alegar nova releitura. A fonte antiga foi preservada em `piper/source/71309a…787e2.py`; o checkpoint público Cadu original G+D permanece como origem do novo piloto.

A correção em `training/v2_piper_experiment.py` usa `torch.no_grad()` na validação. A criação persistente de waveform/espectrograma em `AudioCache` usa `torch.inference_mode(False)` com `no_grad()` e verifica que os tensores resultantes são normais. Fórmulas STFT/mel/loss, fonte Piper, frontend e dados do piloto não mudaram. As sondas de inferência continuam sem gradientes e preservam RNG; elas não criam os caches mel. Os atributos transitórios da atenção/weight_norm são recalculados no forward de treino.

Fonte corrigida: SHA256`2e08e891f89a766eaaaff985c5482cc63d528f199aa453ff9520f985eeb64911`. Oito testes CPU passaram em6,231s, sem carregar modelo completo ou CUDA. Dois novos testes usam **mel_processing original real**, não mock da STFT: reproduzem a falha antiga, aquecem a validação corrigida e completam backward em um novo waveform `requires_grad`; também verificam o cache de áudio criado sob contexto de inferência externo seguido de backward em Conv1d.

Um exercício CPU adicional confirmou Hann/mel`is_inference=false`, gradientes finitos com soma absoluta73,6389 e `cuda_initialized=false`. Receipts: `piper/cpu-tests-cachefix-v2.json` e `piper/cache-autograd-cpu.json`. Nenhum benchmark de produto foi executado nesses testes.

O plano `NEXT-piper-gpu-pilot-v2.json` prepara, sem enfileirar, uma execução nova em `/var/tmp/aegis-v2-work/piper/gpu-engineering-pilot-v2`: freshbase→3pares, processo novo com `--resume …/last.pt --max-updates4`, depois inferência em processo novo. O segundo comando tem uma guarda stdlib testada que exige o sidecar publicado no step3 antes de iniciar a carga; um inicial0 aborta. Deve haver também evento de retomada **step3** antes de aceitar o gate. O código mantém a verificação de SHA de configuração; não migra silenciosamente os estados antigos.

O manifesto `piper/target-native161.json` permanece byte a byte igual ao piloto anterior. A auditoria de `original_text`, caixa e fronteiras G2P por sentença será uma receita científica longa separada. Não usar a correção de autograd como motivo para alterar esse manifesto ou promover o piloto a treino longo. A próxima janela e eventual limpeza são controladas pelo supervisor principal; este agente não acionou CUDA, fila GPU, produção ou commit.
