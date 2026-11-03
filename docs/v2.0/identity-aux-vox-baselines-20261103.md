# Controles Vox no passe auxiliar de identidade

O novo manifest privado preserva os 23 itens anteriores e acrescenta sete
saídas Vox base e sete do LoRA inicial no passo zero: 37 clipes, abaixo do
limite de40. Os papéis são distintos, `generated_language_baseline` e
`generated_target_initialization`; ambos têm zero updates no alvo. Não há
resultado de treino200 neste passe e o teste final permaneceu fechado.

As sete saídas base são rep0 do cliente nativo em streaming compilado,
KV1024/NFE10, sem DSP ou HTTP. O inicial usa NFE10 em síntese eager por
arquivo. Os textos solicitados/efetivos e os sete casos são iguais entre
essas séries; todos os WAVs e recibos foram conferidos por hash. As rotas
streaming/arquivo e compilada/eager continuam fatores de confusão. O novo
manifest é derivado; os23 itens e resultados anteriores não foram alterados.

O helper existente executou encoder Qwen, YIN com máscara de energia e
medidas espectrais/RMS em CPU com duas threads e nice19: 37 clipes em
21,801 s, pico993,81 MiB, CUDA não inicializada. Carregou estritamente apenas
os76 tensores de locutor,17,71 MB de pesos selecionados. Não houve download,
cópia de áudio/pesos nem escuta. O alvo de RMS ativo comum ficou em−23 dBFS,
com ajuste linear só na análise, sem alterar WAVs e sem equivalência LUFS.

| Série de sete casos | Mediana cosseno ao centroide das âncoras | Mediana de F0/YIN por clipe | Centroide espectral mediano |
| --- | ---: | ---: | ---: |
| Vox base nativo KV1024/NFE10 | 0,9168 | 148,0 Hz | 485,4 Hz |
| LoRA inicial passo0, eager/NFE10 | 0,9161 | 151,7 Hz | 469,8 Hz |

A mediana do cosseno entre os pares base/inicial do mesmo texto foi0,9987.
Esses valores apenas registram os controles antes de aprender a Aegis;
não são aprovação, limiar de identidade ou evidência de caráter robótico.
O encoder pertence ao Qwen, as três âncoras dizem textos diferentes e o YIN
não distingue vozeamento de modo confiável. Todas as limitações do helper
ficam no recibo. F0/PSD/cosseno devem acompanhar áudio inédito e clareza.

Artefatos privados:

- `/var/tmp/aegis-v2-work/identity-aux/manifest-vox-baselines37.private.json`, SHA256 `6610c551ab3ab99d5f20e9634e24300c33fb38346cb905b9ca37a0c114a83804`.
- `identity-aux/extend-vox-baselines-20261103.py` e `vox-baselines37-derivation.json`: receita, papéis, hashes e verificação dos23 itens preservados.
- `identity-aux/vox-baselines37/results.json`, `embeddings.private.npz`, `summary.private.json` e `execution-receipt.json`: análise completa, pares dos mesmos textos, custo e espaço.

Não se acrescentarão sete clipes ao37 atual automaticamente: ultrapassaria40.
Quando houver checkpoint200 elegível, um passe novo de38 pode retirar as
seis reconstruções acústicas e incluir suas sete sínteses, conservando três
âncoras, Qwen, F5, Vox base e inicial0. O passe37 fica preservado para auditoria.
Nenhum treinador, helper compartilhado, perfil ou serviço foi alterado.

A receita futura está preparada, sem execução, em
`identity-aux/NEXT-step20038.json` e
`identity-aux/prepare-vox-step20038-20261103.py` (SHA256
`2f88d50dcd472c0f9426f2edac44de4564e6777f99f5e14ab7984deae06f8113`).
Ela exige exatamente um job ASR completo de200, sete casos/NFE10, source e
WAVs com os hashes publicados e um checkpoint do mesmo passo/hash. Conserva
os sete WAVs e esse checkpoint por hardlink numa área própria de identidade,
mesmo se200 não for bestmel ou bestASR. Reserva incremental estimada80 MiB,
com checkpoint limitado a70 MiB, sem carregar modelo ou alterar pesos.
