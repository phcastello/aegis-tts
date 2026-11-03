# Marco400: passe auxiliar e arquivo200 preservado

O manifest novo tem38 clipes, substituindo as sete saídas200 pelas sete400.
Mantém âncoras, Qwen, F5, Vox base e inicial0; o manifest, os áudios e o
checkpoint200 continuam arquivados separadamente. Não foram acumulados45
itens nem aberto o teste final. O checkpoint400 foi conferido por hash e
preservado com os WAVs num snapshot próprio:
`1c9409ddf9e39352618984221d3334e607e3168275cc7585e466d23c409e8c0e`.

As três avaliações ASR nativas usam os mesmos sete textos, eager/NFE10 e
sem DSP. Settings, versões, script, modelo, normalização e expected text
foram conferidos iguais. HTTP original/refinado é outra série, analisada
pelo responsável; não integra estes números.

| Medição | Inicial0 | Passo200 arquivado | Passo400 |
| --- | ---: | ---: | ---: |
| WER verbatim | 22/135 = 16,30% | 16/135 = 11,85% | 15/135 = 11,11% |
| CER verbatim | 51/755 = 6,75% | 36/755 = 4,77% | 30/755 = 3,97% |
| Cosseno mediano ao centroide | 0,9161 | 0,9529 | 0,9831 |
| F0/YIN mediano por clipe | 151,7 Hz | 171,7 Hz | 196,5 Hz |
| Centroide espectral mediano | 469,8 Hz | 604,5 Hz | 788,8 Hz |
| Energia4–8kHz mediana | 0,595% | 2,208% | 3,510% |
| Energia8–12kHz mediana | 0,0837% | 0,2926% | 1,504% |

Como contexto, as três âncoras têm F0/YIN mediano200,1 Hz, centroide897,6 Hz
e energias4–8/8–12kHz de3,416%/2,962%. No mesmo passe38, Qwen processado
apresentou cosseno0,9819 e F5 especializado NFE16 apresentou0,9900.
Essas comparações não são limiar ou aprovação de identidade: o encoder é do
Qwen, as âncoras dizem outros textos, YIN só usa uma máscara de energia e
as bandas não estão alinhadas por fonema. Não houve escuta nem conclusão
sobre caráter metálico, prosódia ou pronúncia.

O helper existente concluiu os38 clipes em19,963 s, CPU2/nice19, pico968,86
MiB e CUDA não inicializada. RMS ativo comum−23 dBFS só na análise; nenhum
áudio original foi alterado. O treino continuou sem intervenção.

Artefatos privados, sob `/var/tmp/aegis-v2-work`:

- `identity-aux/manifest-vox-step40038.private.json`, SHA256 `1a6b24c90d0cceb80a075420536031116e53a583709f6dbfe31f4f3a4ec95d36`.
- `identity-aux/vox-step40038/results.json`, SHA256 `2f2940adc2caf01658d660960f7909edbe6c3867fd08fa2e244ef14a3ed63736`.
- `identity-aux/vox-step40038/{summary.private.json,embeddings.private.npz,execution-receipt.json}`: descritores, pares, ASR0/200/400 e custo/espaço.
- `identity-aux/snapshots/voxcpm2-step400-5453e1f3320a/`: sete WAVs e checkpoint/recibos vinculados; uso definido para reproduzir o marco400 independentemente de bestmel/bestASR.
- `identity-aux/prepare-vox-step40038-20261103.py`: receita única; o200 permanece no diretório próprio anterior.

A retenção desses checkpoints de marcos deve ser revista após uma versão
posterior validada e a decisão final. Não houve alteração de treinador,
helper compartilhado, perfil ou serviço nesta análise.
