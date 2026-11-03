# Marco200: ASR e indícios auxiliares de identidade

O checkpoint200 foi vinculado aos sete WAVs renderizados antes de abrir a
análise: SHA256
`d45d4559e80fbf49c5c6646659be023573067353aaa45cfbee59569168094376`,
54.741.350 bytes. O snapshot próprio por hardlink conserva áudio, checkpoint,
source/jobs/ASR e recibos, mesmo quando o melhor por mel e o melhor por ASR
forem versões diferentes. Nenhum peso foi carregado para inferência nem o
treino foi interrompido nesta tarefa.

A validação ASR do monitor estava completa: sete casos únicos, mesmos textos,
NFE10 eager por arquivo, sem DSP. O controle correto antes/depois é inicial0
versus200; a base compilada em streaming é uma referência separada.

| ASR | Inicial0 | Passo200 |
| --- | ---: | ---: |
| Verbatim/efetivo | 22/135 = 16,30% | 16/135 = 11,85% |
| Spoken | 43/154 = 27,92% | 38/154 = 24,68% |
| CER verbatim/efetivo | 51/755 = 6,75% | 36/755 = 4,77% |
| CER spoken | 123/838 = 14,68% | 114/838 = 13,60% |

Versões, settings, modelo/script ASR, normalização e textos esperados foram
conferidos iguais. CER usa os tokens normalizados unidos por espaços.

O reconhecedor reduziu edições em comando (3→0), pressão (7→4) e sibilantes
(4→2). Resposta curta passou0→1, siglas3→4, pergunta manteve0 e longo5.
As transcrições ainda divergem em14h05, algumas siglas e palavras comuns.
Isso é evidência do reconhecedor, sem afirmar pronúncia ou omissão auditiva.
O timbre pode alterar o próprio ASR; os sete áudios permanecem disponíveis.

O passe novo contém38 clipes: três âncoras e sete de cada série Qwen, F5,
Vox base, inicial0 e200. Retirou só as seis reconstruções acústicas do passe
anterior, preservado. Todos os hashes de texto/áudio coincidiram. O helper
existente rodou CPU2/nice19 por19,622 s, pico979,73 MiB e CUDA não inicializada.
RMS ativo comum−23 dBFS; nenhum WAV original foi modificado ou regravado.

| Série | Cosseno mediano ao centroide | F0/YIN mediano | Centroide espectral | Energia4–8kHz | Energia8–12kHz |
| --- | ---: | ---: | ---: | ---: | ---: |
| Âncoras (3) | — | 200,1 Hz | 897,6 Hz | 3,416% | 2,962% |
| Qwen processado (7) | 0,9819 | 220,3 Hz | 539,1 Hz | 1,290% | 0,803% |
| F5 especializado NFE16 (7) | 0,9900 | 207,3 Hz | 1.152,1 Hz | 4,422% | 4,294% |
| Vox base (7) | 0,9168 | 148,0 Hz | 485,4 Hz | 0,645% | 0,0877% |
| LoRA inicial0 (7) | 0,9161 | 151,7 Hz | 469,8 Hz | 0,595% | 0,0837% |
| LoRA200 (7) | 0,9529 | 171,7 Hz | 604,5 Hz | 2,208% | 0,2926% |

São medianas por clipe. O encoder é do Qwen e não foi calibrado como
verificador; as âncoras dizem outros textos, YIN usa máscara de energia e
as bandas espectrais não estão alinhadas por fonema. A mudança numérica
não aprova identidade, caráter metálico, prosódia ou a voz. Não houve escuta,
limiar automático, teste final ou promoção do modelo.

Artefatos privados conservados:

- `identity-aux/manifest-vox-step20038.private.json`: seleção e hashes.
- `identity-aux/vox-step20038/{results.json,summary.private.json,embeddings.private.npz,execution-receipt.json}`: análise, pares do mesmo texto, ASR antes/depois e ambiente.
- `identity-aux/snapshots/voxcpm2-step200-b877f568ab12/`: sete WAVs, `checkpoint.pt`, sidecar, source/jobs/ASR e `identity-binding.json`.

Todos os caminhos acima partem de `/var/tmp/aegis-v2-work`.
Esse checkpoint tem uso definido para reproduzir o marco de identidade200;
a retenção deve ser revista depois de uma versão posterior validada e da
decisão final. Os insumos e o passe37 continuam preservados.
