# Marco1400: comparação por caso com800 e1000

O checkpoint1400 e os sete WAVs foram vinculados exatamente antes da análise:
`d79948253574f4bd6d6dc8ba7f8247389b852cf6d35d62c5df21b3a3a0ba6e8a`.
O novo manifest conserva38 itens substituindo1000 por1400; todos os snapshots
prévios continuam arquivados. Teste final6 e target50 não foram abertos.

O ASR1400 teve13/135 edições (WER9,63%) e27/755 caracteres (CER3,58%),
comparado a14/135 e31/755 em800;1000 teve16/135 e32/755. O1200 também
tinha13/135 (CER28/755). Settings/modelo/script/textos ASR coincidiram.

A evidência auxiliar de voz é mista: cosseno mediano800→1000→1400 de
0,9884→0,9866→0,9879. A maioria dos casos não aumenta em relação ao800.
F0 mediano entre clipes sobe para226,1 Hz, contra206,4/197,7 Hz em800/1000
e200,1 Hz nas três âncoras. Não há melhora monotônica dos indicadores de voz
acompanhando a perda mel; isso não é um veredito auditivo nem critério automático.

| Caso | Cosseno800 | Cosseno1000 | Cosseno1400 | Faixa entre3 âncoras1400 |
| --- | ---: | ---: | ---: | ---: |
| v_command | 0.9822 | 0.9823 | 0.9817 | 0.9782–0.9817 |
| v_reply | 0.9890 | 0.9863 | 0.9844 | 0.9813–0.9847 |
| v_pressure | 0.9889 | 0.9898 | 0.9879 | 0.9847–0.9878 |
| v_acronyms | 0.9884 | 0.9894 | 0.9884 | 0.9850–0.9886 |
| v_question | 0.9879 | 0.9866 | 0.9880 | 0.9860–0.9874 |
| v_sibilants | 0.9865 | 0.9830 | 0.9851 | 0.9822–0.9851 |
| v_long | 0.9912 | 0.9912 | 0.9903 | 0.9877–0.9899 |

Cada célula abaixo traz F0 p10 / mediana / p90 em Hz:

| Caso | 800 | 1000 | 1400 |
| --- | ---: | ---: | ---: |
| v_command | 80.0 / 208.6 / 305.5 | 116.5 / 206.9 / 293.2 | 79.2 / 233.0 / 257.1 |
| v_reply | 146.1 / 206.0 / 245.9 | 111.2 / 195.6 / 209.2 | 94.7 / 227.0 / 264.1 |
| v_pressure | 90.2 / 226.2 / 287.1 | 88.4 / 206.5 / 276.2 | 82.5 / 226.1 / 262.6 |
| v_acronyms | 101.6 / 199.4 / 262.9 | 83.4 / 197.9 / 260.6 | 89.8 / 230.9 / 295.1 |
| v_question | 155.0 / 206.4 / 230.6 | 98.3 / 190.5 / 214.1 | 102.3 / 209.0 / 275.1 |
| v_sibilants | 66.7 / 187.7 / 238.2 | 68.6 / 174.6 / 207.9 | 68.2 / 198.9 / 261.3 |
| v_long | 88.1 / 209.0 / 276.4 | 95.2 / 197.7 / 263.4 | 88.7 / 207.2 / 258.7 |

No1400, os medianos F0 por clipe variam198,9–233,0 Hz (DP descritivo12,5 Hz).
A largura interna p10–p90 varia169,5–205,3 Hz (mediana177,9 Hz). Centroide
espectral mediano940,9 Hz; energia4–8kHz4,417% e8–12kHz2,528%.

O encoder é do Qwen, as três âncoras dizem outros textos e YIN usa máscara
de energia, sujeita a quadros não vozeados e erros de oitava. Faixas não são
intervalos de confiança. Não houve escuta, limiar ou aprovação de identidade.
Os arquivos800/1000 foram comparados somente após conferir o mesmo encoder,
código de análise e política de ganho/RMS. As séries são nativas eager/NFE10,
sem DSP; resultados HTTP/frontend numérico permanecem separados.

A análise CPU2/nice19 levou19,687 s, pico1010,19 MiB, CUDA não inicializada.
Não houve inferência TTS, mudança de código compartilhado ou interrupção.

Artefatos sob `/var/tmp/aegis-v2-work`:

- `identity-aux/manifest-vox-step140038.private.json`: seleção38 e hashes.
- `identity-aux/vox-step140038/{results.json,summary.private.json,embeddings.private.npz,execution-receipt.json}`.
- `identity-aux/snapshots/voxcpm2-step1400-8da137fdd78b/`: sete WAVs, CP1400, sidecars e fontes.
- `identity-aux/prepare-vox-step140038-20261103.py`: receita do passe.

Após este marco, o revisor fica inativo; não foi criado polling de IA por200updates.
