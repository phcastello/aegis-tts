# Identidade auxiliar: dois controles de outras vozes

Dois controles existentes e previamente fixados foram acrescentados ao passe
de identidade para a mesma frase pública de validação `v_reply`.
Nenhum áudio, TTS ou ASR novo foi produzido.

O primeiro controle é Piper Cadu BASE, síntese nativa no passo0, antes de
qualquer atualização com a Aegis. O segundo é FastPitch público no passo5500,
com BigVGAN, sem treino nos dados da Aegis. A distinção de locutor vem da
proveniência dos modelos; não houve avaliação auditiva desses clipes.

O helper `training/v2_identity_aux.py`, SHA256
`c9cc80bdd2ccad2b49e17b56a17b28d3b28907dd7fbdc19cf1aeb1edbbbc663f`,
analisou40 itens em CPU:21,35s e998MiB de RSS máximo, com CUDA não inicializada.
Reutilizou somente o encoder local já auditado. As38 linhas anteriores,
35 resultados de cosseno, pares de âncoras, encoder e política de normalização
permaneceram idênticos. O alvo de RMS ativo continuou em−23dBFS; isso não é LUFS.

| Sistema, mesma frase `v_reply` | Cosseno com centroide das âncoras |
| --- | ---: |
| Piper Cadu BASE, passo0 |0,924987 |
| FastPitch público5500 |0,932038 |
| Qwen produção |0,979064 |
| F5 especializado, NFE16 |0,988503 |
| VoxCPM2, passo1800 |0,985945 |
| VoxCPM2, passo2800 |0,983411 |
| VoxCPM2, passo4000 |0,984213 |

O encoder separou estes dois controles das cinco saídas dos sistemas Aegis
nesta frase. Essa observação é local: duas vozes e uma frase não calibram um
verificador populacional, um limiar de aceitação ou a identidade robótica.
O passe não elege vencedora.

O F0 mediano estimado por YIN foi aproximadamente115/143Hz nos controles e
180–220Hz nas saídas dos sistemas Aegis. Registro e prosódia podem contribuir
para a separação; essa medição não isola caráter metálico. Permanecem os
vieses do encoder pertencente ao Qwen e os textos diferentes das âncoras.
A identidade requer escuta pareada junto das verificações de clareza.

Provas locais, preservadas separadamente dos resultados anteriores:

- `/var/tmp/aegis-v2-work/identity-aux/negative-controls-20261103/REPORT.private.json` — SHA256 `5aa028e46f76e06c7da7a757a42484f8eb97e1ca49b9a2c57a4933e8e543511a`.
- `/var/tmp/aegis-v2-work/identity-aux/negative-controls-20261103/comparison.private.json` — SHA256 `b1b5d9be77868963c3b9a25fd457fb432f33d37873d2cc27c4e8a48059097657`.
- `/var/tmp/aegis-v2-work/identity-aux/negative-controls-20261103/analysis40/results.json` — SHA256 `23a8a67853541a757ff955ff775a3be671df5bb86333762d7bb5266d793ab759`.
