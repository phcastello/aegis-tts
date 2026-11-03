# Comparação inicial local, 03/11/2026

Medições na mesma A2000, validação comum e regra digital congelada. Qwen inclui gateway e DSP; F5/VITS medem motor e arquivos, incluindo conversões e fechamento. Não há transporte HTTP hipotético nos motores. Os modelos e repetições falhos ficam no denominador. Esta tabela não aprova identidade, pronúncia ou uma candidata para produção.

| Caso | Qwen TTFA p50 s | F5 NFE8 TTFA p50 s | F5 NFE16 TTFA p50 s | VITS base TTFA p50 s |
| --- | ---: | ---: | ---: | ---: |
| v_command | 1.844 | 0.810 | 1.878 | 0.209 |
| v_reply | 1.911 | 0.807 | 1.713 | 0.181 |
| v_pressure | 1.813 | 1.030 | 2.583 | 0.194 |
| v_acronyms | 1.852 | 1.480 (n=2) | 2.457 | 0.191 |
| v_question | 1.804 | 0.917 (n=2) | 2.353 | 0.215 |
| v_sibilants | 1.800 | 0.906 | 2.513 | 0.210 |
| v_long | 1.828 | 1.035 | 2.708 | 0.255 |

P95 indisponível com n=3 por caso. TTFA é reprodução FIFO estimada até início digital de fala, sem atraso do dispositivo. Ver também a disponibilidade dos samples de fala nos recibos, sem confundir as duas métricas.

| Sistema | PCM válidos / tentativas | RTF mediano observado* | Máximo de memória observado |
| --- | ---: | ---: | --- |
| qwen-warm | 21 / 21 | 0.686 | 10.674 MiB dispositivo, amostrado |
| f5-nfe8 | 19 / 21 | 0.133 | 2124.5 MiB allocator PyTorch |
| f5-nfe16 | 21 / 21 | 0.263 | 2130.9 MiB allocator PyTorch |
| vits-base | 21 / 21 | 0.029 | 489.4 MiB allocator PyTorch |

*A mediana RTF desta linha mistura os sete tamanhos e serve só como descrição; decisões usam cada frase. Picos de allocator e de dispositivo têm escopos diferentes. Não comparar diretamente os números de memória entre esses escopos.

F5 NFE8 teve duas conversões rejeitadas: pergunta/r0, pico FLOAT 1,14484 com 22 samples fora da faixa; siglas/r2, pico 1,03475 com um sample fora. Não houve clipping oculto. O ASR da pergunta usa FLOAT bruto e fica em coorte diagnóstica separada, sem TTFA de produto. F5 NFE16 completou 21/21 PCM, mas não alcançou o ganho material selado de 400 ms em nenhum caso. VITS base é um controle de idioma/velocidade antes de especializar a identidade.

## ASR auxiliar na repetição zero

| Sistema / coorte | Casos | Edições/palavras verbatim | WER auxiliar | CER auxiliar |
| --- | ---: | ---: | ---: | ---: |
| bigvgan-roundtrip / reserved-vocoder-roundtrip | 3 | 0/73 | 0.00% | 0.00% |
| f5-specialized-nfe16 / validation-rep0 | 7 | 26/135 | 19.26% | 7.55% |
| f5-specialized-nfe8 / validation-rep0 | 6 | 26/120 | 21.67% | 10.15% |
| f5-specialized-nfe8 / validation-rep0-PCM-rejected | 1 | 5/15 | 33.33% | 14.67% |
| qwen-production / validation-rep0 | 7 | 21/135 | 15.56% | 7.28% |
| target-source / reserved-vocoder-roundtrip | 3 | 0/73 | 0.00% | 0.00% |
| vits-cv-pt-base / validation-rep0 | 7 | 50/135 | 37.04% | 16.56% |
| vits-posterior-reconstruction / reserved-vits-reconstruction | 3 | 9/84 | 10.71% | 5.96% |
| vits-target-source / reserved-vits-reconstruction | 3 | 2/84 | 2.38% | 2.98% |

Comparar percentuais somente com os mesmos casos. As formas por extenso e as entradas efetivas de cada frontend permanecem separadas no recibo privado; números/siglas escritos de outro modo podem aumentar WER sem indicar pronúncia errada. ASR também pode errar com o timbre robótico. Não foi feita avaliação auditiva. Um sotaque português distinto não deve ser tratado como falha do modelo apenas por esse reconhecedor.

Os sinais de omissão, inserção, última palavra e n-gramas repetidos são alinhamentos de transcrição para revisão, sem afirmar que o áudio contém esses defeitos. Detalhes privados: `/var/tmp/aegis-v2-work/asr/initial-comparison.json`. Fontes: `/var/tmp/aegis-v2-work/bench`, `/var/tmp/aegis-v2-work/asr/initial-candidates.json`, `/var/tmp/aegis-v2-work/asr/qwen-warm.json`.

Os três roundtrips BigVGAN e os três VITS posterior→decoder têm textos e alvos diferentes da validação inédita e entre si. Seus controles de fonte estão na mesma coorte correspondente; não misturar esses percentuais com a síntese texto→fala nem entre codecs.
