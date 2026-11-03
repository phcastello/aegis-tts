# Marco1000: intervalos por clipe e dispersão de F0

O ASR1000 estava completo, mas seu checkpoint já não estava retido pelo
monitor de ASR. O `last.pt` era1044 e não foi usado. A cópia foi vinculada
pelo `best.pt` de passo1000 e pelo SHA256 exato:
`3a8c006a0ff980d78f29d943468bdf2dca6801d9e1e2f60ce4a023da583c06e3`.
Sete WAVs, checkpoint, source e recibos ficaram num snapshot próprio.
O manifest novo substitui800 por1000, conservando38 itens e todos os
arquivos anteriores. Nenhum teste final foi aberto.

São os mesmos sete textos nativos, eager/NFE10 e sem DSP. O ASR1000 teve
16/135 edições (WER11,85%) e32/755 caracteres (CER4,24%). O800 tinha14/135
(10,37%) e31/755 (4,11%). Cosseno mediano1000 de0,9866 também não supera o
0,9884 do800. Essas oscilações ficam registradas, sem selecionar por mel ou
por um único indicador. Settings, versões e textos ASR foram conferidos iguais.

| Caso1000 | Cosseno ao centroide | Min–max entre3 âncoras | F0 p10 / mediana / p90, Hz |
| --- | ---: | ---: | ---: |
| v_command | 0,9823 | 0,9795–0,9823 | 116,5 / 206,9 / 293,2 |
| v_reply | 0,9863 | 0,9843–0,9860 | 111,2 / 195,6 / 209,2 |
| v_pressure | 0,9898 | 0,9872–0,9897 | 88,4 / 206,5 / 276,2 |
| v_acronyms | 0,9894 | 0,9871–0,9893 | 83,4 / 197,9 / 260,6 |
| v_question | 0,9866 | 0,9849–0,9863 | 98,3 / 190,5 / 214,1 |
| v_sibilants | 0,9830 | 0,9818–0,9819 | 68,6 / 174,6 / 207,9 |
| v_long | 0,9912 | 0,9888–0,9913 | 95,2 / 197,7 / 263,4 |

Entre os sete clipes1000, o cosseno ao centroide varia0,9823–0,9912 e os
F0 medianos variam174,6–206,9 Hz, com desvio descritivo10,2 Hz. A largura
interna p10–p90 varia98,0–187,8 Hz (mediana168,2 Hz). Não são intervalos de
confiança. O YIN usa máscara de energia, com possíveis erros de oitava e
quadros não vozeados; essa dispersão não prova instabilidade/prosódia auditiva.

| Contexto no mesmo passe | Cosseno mediano (min–max de7 clipes) | F0 mediano entre clipes, min–max | Largura interna p10–p90 mediana |
| --- | ---: | ---: | ---: |
| Qwen processado | 0,9819 (0,9779–0,9850) | 208,6–246,6 Hz | 80,7 Hz |
| F5 especializado NFE16 | 0,9900 (0,9867–0,9928) | 192,8–209,5 Hz | 161,9 Hz |
| LoRA inicial0 | 0,9161 (0,8939–0,9230) | 120,2–187,4 Hz | 97,8 Hz |
| LoRA1000 | 0,9866 (0,9823–0,9912) | 174,6–206,9 Hz | 168,2 Hz |

As três âncoras têm F0 mediano entre195,4–201,4 Hz e largura interna mediana
152,9 Hz. O encoder é do próprio Qwen e as âncoras dizem outros textos.
Não houve escuta, limiar de identidade ou conclusão de superioridade vocal.
O sumário privado inclui os três cossenos individuais e estatísticas por
clipe de todas as séries, evitando uma decisão apenas pela mediana.

A análise CPU2/nice19 levou19,973 s, pico990,64 MiB e CUDA não inicializada.
Nenhuma inferência TTS, mudança de treinador ou interrupção foi feita.

Artefatos privados sob `/var/tmp/aegis-v2-work`:

- `identity-aux/manifest-vox-step100038.private.json`, SHA256 `1a62807688f1504fe5fc9aaf5708a80a8d1546d84b1e9c59d5440bb2e173af2b`.
- `identity-aux/vox-step100038/results.json`, SHA256 `9dc9958ab00c5ce0cc68861d1cd046c6109495f8faacc1eb6dc222b0fdbf99c5`.
- `identity-aux/vox-step100038/{summary.private.json,embeddings.private.npz,execution-receipt.json}`.
- `identity-aux/snapshots/voxcpm2-step1000-e9b2970bb344/`: WAVs e checkpoint1000 exatos, independentes de futuras trocas de bestmel/bestASR.

Depois deste passe, não foi programada consulta de IA por200updates. O
monitor local continua registrando ASR; a próxima revisão geral cabe ao
responsável na janela prevista.
