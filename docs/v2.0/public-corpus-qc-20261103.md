# TTS-Portuguese: dados usados e QC local, 03/11/2026

O corpus público fornece uma rota de adaptação do idioma antes de especializar
as quatro horas privadas da Aegis. A documentação dos autores descreve cerca de
**10 h 28 min** nominais de um locutor, 48 kHz, com textos de Wikipedia,
conversação e frases balanceadas. Não é um corpus de muitos locutores ou uma
amostra demonstrada de todos os sotaques brasileiros.
[Fonte dos autores fixada](https://github.com/Edresson/TTS-Portuguese-Corpus/blob/664b3d574616cabb477f9e7a95811d65b5e8c968/README.md).

Licença **CC BY 4.0**, com atribuição e indicação das mudanças na distribuição.
Autores: Edresson Casanova, Arnaldo Candido Junior, Christopher Shulby, Frederico
Santos de Oliveira, João Paulo Teixeira, Moacir Antonelli Ponti e Sandra Maria
Aluisio. A licença do corpus não transfere direitos sobre a voz privada Aegis.
[Licença publicada](https://github.com/Edresson/TTS-Portuguese-Corpus/blob/664b3d574616cabb477f9e7a95811d65b5e8c968/LICENSE).

Fonte local: `/var/tmp/aegis-v2-work/portuguese/`; revisão upstream
`664b3d574616cabb477f9e7a95811d65b5e8c968`. ZIP obtido no link Dropbox oficial,
2.371.059.638 bytes, SHA-256
`9704531ca602bdef8d974fb440decd104e37db32e5c88a10c53fa15da580966f`.
O arquivo efetivamente obtido tem 3.625 linhas e 3.624 WAVs presentes; falta
`sample-848`. As contagens reais prevalecem sobre o total nominal do artigo.

Preservou-se o ZIP original. Foram selecionadas falas completas de 1–22 s,
mono, finitas e sem clipping, sem denoise, corte, ganho, alteração de pitch ou
velocidade. Preparação 48→24 kHz por `resample_poly`, Kaiser beta 10, PCM24;
partições agrupam lotes de origem de 50 IDs e componentes com texto/áudio
idênticos antes do sorteio fixo. Estas são mudanças documentadas, não gravações
novas dos autores. A atribuição completa e as mudanças ficam em
`portuguese/ATTRIBUTION.txt`, com cópias do README e da licença upstream.

| Seleção real | Treino | Validação | Teste reservado* |
| --- | ---: | ---: | ---: |
| Preparação inicial | 3.044 / 8,074922 h | 174 / 0,414256 h | 177 / 0,504197 h |
| FastPitch após excluir transcrição ambígua `sample-828` | **3.043 / 8,073747 h** | 174 / 0,414256 h | 177 / 0,504197 h |
| VITS após quarentena adicional de frontend | 3.026 / 8,012392 h | 172 / 0,405361 h | 177 / 0,504197 h |

*Contagens do manifesto de seleção já registrado; o QC abaixo não abriu nem
transcreveu arquivos do teste. As receitas devem registrar a seleção específica
usada em cada checkpoint. Não apresentar 10 h 28 min nominais como horas treinadas.

A cobertura escrita da seleção FastPitch tem 11.188 palavras distintas e
56.006 ocorrências, incluindo `ã` 2.092, `õ` 438, `ç` 1.806, `nh` 556, `lh` 513,
`rr` 467 e `ss` 1.177. São indícios ortográficos; não são rótulos fonéticos ou
prova de realização correta de nasalização, acentuação e sibilantes. Há poucos
algarismos porque números são normalmente escritos por extenso. Siglas, nomes
Aegis e vocabulário de operação continuam exigindo validação específica em
síntese inédita, antes e depois da adaptação da voz.

## QC ASR antes do treino longo

A amostra foi fixada **antes da transcrição**: uma fala de cada quartil de
duração do treino e da validação, oito no total. Dentro de cada quartil,
seleção determinística priorizou nasalização escrita, diacríticos e números,
com desempate por hash de ID. Nenhum recorte, correção de transcript ou escolha
pela saída ASR. Os WAVs preparados foram conferidos contra seus hashes.

| Split / quartil | ID | Duração s | Edições / palavras esperadas | Observação auxiliar |
| --- | --- | ---: | ---: | --- |
| treino / 1 | sample-215 | 4,74 | 4 / 17 | divergências em touros/verbo/artigo; revisar áudio |
| treino / 2 | sample-79 | 8,39 | 8 / 26 | números e percentual escritos com algarismos pelo ASR |
| treino / 3 | sample-4253 | 11,46 | 0 / 30 | igualdade textual normalizada |
| treino / 4 | sample-4684 | 15,81 | 1 / 28 | cem → 100 na transcrição ASR |
| validação / 1 | sample-5713 | 2,93 | 1 / 7 | Faz → Paz no ASR; revisar áudio |
| validação / 2 | sample-1446 | 5,53 | 3 / 14 | decimal por extenso → algarismos |
| validação / 3 | sample-1076 | 10,37 | 12 / 30 | anos e percentual escritos com algarismos |
| validação / 4 | sample-1085 | 18,65 | 3 / 33 | junção e/terminam/como divergente; revisar áudio |

WER auxiliar agregado: treino **13/101 = 12,87%**, CER 8,45%; validação
**19/84 = 22,62%**, CER 16,38%. As referências `verbatim` e `effective`
(NFC e tipografia normalizada, sem mudar palavras) tiveram métricas iguais.
Não se reescreveu a hipótese ASR para baixar o score. **24 das 32 edições**
concentram-se nos quatro casos numéricos, com forte efeito de representação
por extenso/algarismos. Esses percentuais não demonstram 32 palavras
pronunciadas incorretamente nem permitem corrigir automaticamente o corpus.

Auditor idêntico ao da comparação: `Systran/faster-whisper-small`, revisão
`536b0662742c02347bc0e980a01041f333bce120`, arquivos revalidados; CPU int8,
duas threads, um worker, nice 19, idioma pt, beam 5, temperatura zero, sem
prompt, VAD ou condicionamento no texto anterior. ASR local, sem envio externo.
Resample auxiliar para 16 kHz/Kaiser10, sem ganho ou corte. Não houve escuta.
As divergências comuns não foram tratadas como erros certos de sotaque ou
fonética nem usadas para reescrever transcrições sem revisão.

O QC não encontrou uma quebra global óbvia da correspondência áudio/texto
nessa pequena amostra, e oferece controles para o estágio de idioma. A
aprovação de pronúncia PT-BR permanece dependente das frases inéditas
congeladas, números/siglas, revisão dos divergentes e eventual escuta. Um modelo
com WER pior ou timbre estranho não deve ter sua falha atribuída ao corpus
somente com esses oito pontos.

Recibos: `/var/tmp/aegis-v2-work/asr/public-qc/`. `selection.json` fixa fonte,
partição, quartil, critérios e hashes, SHA-256
`b7c1ab2a3395818cb958f86def6c8b67975e6dbe11cc61c8e90f348f5cc3f913`.
`receipt.json` conserva hipóteses, segmentos e scores, SHA-256
`ca84ff7a4fb6c6912b01c08e84df6f37d856a494245715ad2f97554539b2bc48`.
A seleção dos dados e o QC podem ser repetidos com `jobs.json` e
`benchmarks/v2/asr_score.py` no ambiente F5 documentado.
