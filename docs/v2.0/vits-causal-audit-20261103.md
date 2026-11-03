# Auditoria da regressão VITS após a primeira época — 03/11/2026

Não foi encontrado erro concreto de batch, IDs, extração acústica ou caminho
de gradiente que explique a regressão. A melhoria da reconstrução posterior
de0,613 para0,471 não aprova o prior texto→fala. A hipótese atual é adaptação
instável/esquecimento sob microbatch1; a ablação de ruído e novos controles
precisam decidir isso. A linha não foi descartada por uma época.

Não houve GPU nesta auditoria. Dados, receita executada e checkpoints
permaneceram intactos. Artefatos preservados: best3026/época1 SHA
`da34be18f34182f2c1500754f263628e4baf16179fe68fa743427307cb2de62a`;
last3359/época2 SHA
`a48a5cd0647f38f6d4a09d3379c81702d51782d7667378cd3d85161d94c982c7`.

## Provas executadas

- O construtor local foi comparado ao `VitsDataset.collate_fn` oficial,
  `format_batch` e `format_batch_on_device`, sobre o mesmo áudio público
  real e texto efetivo. **Igualdade bit a bit** em tokens, token lengths,
  waveform, speaker IDs, language IDs, spec, mel, spec lengths e mel lengths.
  Speaker=`male/0`, idioma=`pt/0`, como nos mapas da base.
- O `VitsDataset` oficial carrega diretamente áudio normalizado PCM por
  torchaudio; não aplica o AudioProcessor de outros modelos. Não há ganho,
  trim ou normalização de loudness faltante do pipeline VITS. A diferença
  explícita é o resampling Kaiser10 para22.050Hz a partir do input auditado.
- As funções de spec/mel0.8 foram executadas sobre um segundo real, com
  somente bridges de API `return_complex=False` e argumentos nomeados
  librosa. Comparação com0.28: **spec e mel iguais bit a bit, erro0,0**.
-3.198 falas treino/val foram conferidas quanto a frames/token: nenhuma
  impossível para MAS, mínimo1,66frame/token, mediana3,64. Isso não valida
  cada palavra das transcrições; permanece a incerteza já registrada828.
- Backward real em CPU do modelo PT e losses oficiais, numa fala pública:
  lossKL3,7308; duração1,7879; mel ponderada16,6496; gerador total28,1899.
  Grads finitos, não nulos em todos os componentes relevantes. NormasL2:
  text encoder10,04; SDP3,64; flow13,00; posterior107,50; decoder342,13.
  O critic usado nesse diagnóstico foi novo; sua loss adversarial não deve
  ser comparada numericamente à do critic treinado. Nenhum update foi salvo.
- G/D, KL, duração estocástica, speaker conditioning e detach do SDP usam
  funções oficiais. A ordem D→G e o congelamento temporário do critic no
  backwardG preservam o caminho através da waveform. Não houve reinit do
  text encoder, SDP ou embeddings durante a execução.
- O treino foi FP32, AdamW2e-5 para G/D, betas(.8,.99), eps1e-9; scheduler
  avança por época. A retomada anterior preservou estados e não identificou
  NaN/OOM. Não há evidência de overflow AMP nesta execução FP32.

Recibos privados:
`/var/tmp/aegis-v2-work/vits/causal-audit/{result,legacy-mel,lengths}.json`.
O diagnóstico principal é reproduzível por `training.v2_vits_causal_audit`,
sem GPU. O comparativo legacy informa explicitamente os bridges de API;
não simula execução integral do ambiente antigo.

## Mudanças efetivas do modelo

Drift relativoL2 após3026updates: text encoder2,0%, SDP1,4%, decoder1,9%,
posterior1,6%, flow2,0%, embeddings0,6%. Isso demonstra adaptação real,
sem estabelecer causalidade perceptiva.

Com ruído de duração zero, mesmos tokens/seed:

| Texto de validação | Base, segundos preditos | PT época1 |
| --- | ---: | ---: |
| Comando |2,57 |2,25 |
| Resposta |2,40 |1,95 |
| Pressão |6,63 |5,58 |
| Siglas |5,17 |4,68 |
| Pergunta |5,47 |4,67 |
| Sibilantes |5,13 |4,08 |
| Longo |23,32 |19,25 |

O prior ficou sistematicamente mais rápido. O log de sigma médio mudou
aproximadamente−0,68→−0,47, aumento de cerca de24% em sigma. Portanto,
ruído nominal0,3 não representa a mesma perturbação acústica absoluta antes
e depois do treino. Os arquivos reais0/0 e0,3/0,3 devem separar contribuição
estocástica de alteração do prior. Duração predita não é avaliação auditiva.

## Próximos controles propostos, sem execução

1. Finalizar a ablação já planejada base/PT×ruído0/0 ou0,3/0,3, com o mesmo
   checkpoint e textos. Não abrir o teste final. Isso precede dias adicionais.
2. Se a regressão persistir nas duas políticas, iniciar **experimento novo
   da base original**, preservando esta tentativa: G-LR2e-6 em vez de2e-5,
   D-LR2e-5, mesma seleção e microbatch1. Sondas nos250/500updates mostram
   cedo se há esquecimento, sem esperar outra época inteira.
3. Um segundo contraste pode usar acumulação8 e LR original, comparando após
   a mesma quantidade de falas processadas. São fatores separados; alterar
   LR e batch juntos não identifica o responsável.
4. Congelar text encoder/embedding de idioma e SDP é um **controle de
   preservação da pronúncia**, especialmente útil para futura identidade.
   Não anunciar isso como aprendizado de PT-BR: pode bloquear adaptações de
   idioma necessárias. Descongelamento posterior, com LR menor e validação
   textual, só faz sentido se o controle mantiver clareza.

Decisões devem acompanhar omissões, repetições, negações, sibilantes e caudas
do texto longo, além da métrica ASR fixa. Loss posterior menor isolada não
justifica continuar; uma hipótese de estabilidade diferente ainda é viável.
