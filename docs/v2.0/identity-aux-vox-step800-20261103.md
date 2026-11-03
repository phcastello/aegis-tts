# Marco800: identidade auxiliar em38 clipes

O novo passe substitui as sete saídas400 pelas sete800, conservando38 itens.
Âncoras, Qwen, F5, Vox base e inicial0 permanecem; arquivos e snapshots200/400
não foram alterados. O checkpoint800 e seus sete WAVs estão ligados ao mesmo
source/step/hash, sem usar o último checkpoint845 como substituto:
`b9160041e0773799ab20e2066fb526f7151425bb8055491add375b1007501f33`.

| Medição nativa, eager/NFE10 | Inicial0 | Passo200 | Passo400 | Passo800 |
| --- | ---: | ---: | ---: | ---: |
| WER verbatim | 22/135 = 16,30% | 16/135 = 11,85% | 15/135 = 11,11% | 14/135 = 10,37% |
| CER verbatim | 51/755 = 6,75% | 36/755 = 4,77% | 30/755 = 3,97% | 31/755 = 4,11% |
| Cosseno mediano ao centroide | 0,9161 | 0,9529 | 0,9831 | 0,9884 |
| F0/YIN mediano por clipe | 151,7 Hz | 171,7 Hz | 196,5 Hz | 206,4 Hz |
| Centroide espectral mediano | 469,8 Hz | 604,5 Hz | 788,8 Hz | 904,6 Hz |
| Energia4–8kHz mediana | 0,595% | 2,208% | 3,510% | 3,654% |
| Energia8–12kHz mediana | 0,0837% | 0,2926% | 1,504% | 2,385% |

O passo600, sem passe de identidade próprio aqui, teve17/135 no ASR. Essa
oscilação e o pequeno aumento de CER400→800 ficam registrados; não foram
omitidos para apresentar melhora monotônica. São os mesmos sete textos e
settings/versões/modelo/script/normalização ASR conferidos iguais. As medições
HTTP e o frontend numérico pertencem a séries separadas.

Como contexto, as âncoras têm F0/YIN200,1 Hz, centroide897,6 Hz e energia nas
duas bandas de3,416%/2,962%. O mesmo passe traz cosseno0,9819 no Qwen e0,9900
no F5 especializado NFE16. Cosseno não é limiar nem aprovação: o encoder é
do Qwen, âncoras dizem outros textos, YIN usa máscara de energia e PSD não
está alinhada por fonema. Não houve escuta ou conclusão de superioridade vocal.

A análise usou somente CPU2/nice19:19,790 s, pico981,24 MiB e CUDA não
inicializada. RMS ativo comum−23 dBFS só na análise. Nenhuma inferência TTS,
interrupção do treino, mudança de runtime ou acesso ao teste final ocorreu.

Artefatos privados sob `/var/tmp/aegis-v2-work`:

- `identity-aux/manifest-vox-step80038.private.json`, SHA256 `dd9713e47eef61b0ecccf0c177b3e69f14af9b130698d60978943f322fc8446e`.
- `identity-aux/vox-step80038/results.json`, SHA256 `7f0b7707b5c65de9b28eea3086a5627dac390f90e20fc5de5c1bba69b6650daf`.
- `identity-aux/vox-step80038/{summary.private.json,embeddings.private.npz,execution-receipt.json}`: comparações e custo/espaço.
- `identity-aux/snapshots/voxcpm2-step800-78680488bcf4/`: sete WAVs, checkpoint800, sidecar e fontes ASR/metadados.
- `identity-aux/prepare-vox-step80038-20261103.py`: receita do passe, sem acumular45 clipes.
