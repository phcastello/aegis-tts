# Dataset PT-BR: uma operação no Melodyne

Estado em 01/11/2026: o pipeline e os testes sintéticos estão preparados.
**Nenhuma geração de voz, compra, download de checkpoint, ajuste ou treino foi executado.**
A produção TTS não usa este pipeline. A referência pessoal protegida não foi
reutilizada. Não existe `master_raw.wav` real nem `master_processed.wav` tratado.
A geração de material para Melodyne depende de revisão/aprovação do proprietário.

O único master disponível é uma demonstração de **tons sintéticos, sem fala**, em
`/var/tmp/aegis-tts-dataset-20261101/synthetic-demo/master_raw.wav`, com manifesto
no mesmo diretório. Ela dura 22,8 s; SHA256
`a0ace3e5592e69ad289da70b249aeab53aa327673518835a00bc58a887826978`.
`master_processed.wav` nessa demonstração é somente uma cópia para testar recorte,
não uma receita de efeito nem material de treinamento. Todos seus registros
recebem `synthetic_fixture=true` e `content_review=rejected`.

## Corpus e separação

`dataset/corpus_source.txt` contém 132 frases originais em onze categorias:
cotidiano, perguntas, números/horários/datas/medidas, termos técnicos, negativas,
confirmações, personalidade Aegis, natureza, cultura, nomes/pronúncia e extensões (frases muito curtas e longas).
`dataset/corpus.pt-BR.jsonl` contém o piloto e uma expansão **rascunho** de
792 registros. A expansão usa contextos discursivos controlados; precisa de
curadoria antes de gerar, e não é uma garantia de variedade ou qualidade final.
Não se deve pagar para sintetizar todo o rascunho antes de aprovar o piloto.

IDs derivam do conteúdo, e variantes próximas compartilham `group`/`group_id`.
Os splits train/validation/test são definidos por hash do grupo, antes do treino.
Mudar o texto cria um ID novo. Não mover variantes entre splits. A regeneração
com `python3 scripts/dataset_corpus.py` é determinística. O hash do corpus
selecionado aparece no dry-run e deve corresponder ao corpus aprovado.

A 150 palavras por minuto, o piloto estima 14,0 minutos e a expansão 118,1
minutos. Esses valores são planejamento, não duração medida nem garantia
científica. O piloto seleciona 112 exemplos, incluindo oito frases muito curtas e quatro acima de quarenta palavras. A geração real medirá duração por clipe e o manifesto distinguirá
áudio dos clipes, faixa entre primeira/última atividade e silêncio separador.
A primeira versão completa deve visar 1–2 horas úteis; expandir depois somente
se erros reais justificarem. Substituir variantes fracas por textos novos antes
de investir nessa expansão.

## Ambiente e configuração explícita

Use Python 3.12 em ambiente separado; o container de produção não recebe novas
dependências. `dataset/requirements.txt` fixa numpy/scipy nas versões já testadas. A montagem mantém somente um clipe decodificado por vez; a importação libera o master bruto antes de carregar o tratado. Um preflight recusa WAV cujo working set estimado supere 8 GiB ou a RAM livre do host, antes da decodificação; exportações longas a 192 kHz exigem reexportar a 24 kHz ou implementar processamento por janelas.
Arquivos gerados são aceitos somente fora do checkout. Diretórios privados usam
0700 e os arquivos 0600, incluindo manifesto, transcrições e revisão.

```bash
python3 -m venv /var/tmp/aegis-dataset-venv
/var/tmp/aegis-dataset-venv/bin/pip install -r dataset/requirements.txt
python3 scripts/dataset_pipeline.py generate \
  --config dataset/provider.example.json \
  --work-dir /var/tmp/aegis-tts-dataset-20261101/pilot --dry-run
```

No host não há configuração de dataset. Estão ausentes:
`provider`, `base_url`, `model`, `voice`, `terms_url`, `terms_reviewed_on`,
`price_per_1000_characters`, `max_cost`, `max_characters`, `max_requests`,
`max_examples`, `training_rights_confirmed=true`, `voice_rights_confirmed=true`,
`generation_approved=true`, `approval_reference` e `corpus_review_sha256`.
`currency` está definido apenas como unidade de estimativa, USD.

O adaptador preparado suporta `provider=openai-compatible-http`, com `base_url`
apontando ao **endpoint completo de síntese**, não à raiz da API. A voz e o modelo
são obrigatórios e seguem em cada pedido. Não há fallback para outra voz, para
o serviço local protegido ou para a referência do proprietário. Se houver chave,
configure apenas o nome da variável em `api_key_env`; o segredo fica no ambiente.
Não colocar chaves na configuração, no código ou em comandos versionados.

Antes de habilitar geração, verificar os termos atuais da API e da voz escolhida,
incluindo uso do áudio em treinamento e direito de usar a voz. A URL e a data
ficam na proveniência. **Nenhum provedor foi escolhido ou teve direitos de treino
confirmados nesta entrega.** Os booleanos registram a revisão; o script não faz
uma análise jurídica automática. Preço por mil caracteres é uma regra explícita
do adaptador: se o provedor cobrar por tokens, segundos ou outra unidade, adaptar
o cálculo antes da geração. Não assumir que essa aproximação permite exceder o teto.

Dry-run não envia pedidos. Com a configuração vazia, retorna 112 exemplos,
12.736 caracteres, custo indeterminado e todos os campos faltantes. O SHA256 do
piloto atual é `60e5ac8ea6b452101f18f97ebb80b441bf5e48c7f5fedbb4aa85479f3756da1f`.
A aprovação de geração deve identificar a revisão e esse hash. Não preencher
campos de aprovação por inferência.

## Geração retomável e montagem futura

**Os comandos a seguir são um procedimento futuro, após revisão e aprovação.**
Copiar `provider.example.json` para um caminho privado e preencher todos os
campos aprovados. Começar pelo piloto; `--stage full` é uma etapa posterior.

```bash
python3 scripts/dataset_pipeline.py generate \
  --config /CAMINHO_PRIVADO/dataset-provider.json \
  --work-dir /CAMINHO_PRIVADO/pilot --stage pilot
python3 scripts/dataset_pipeline.py assemble \
  --config /CAMINHO_PRIVADO/dataset-provider.json \
  --work-dir /CAMINHO_PRIVADO/pilot --stage pilot
```

A geração usa concorrência de 1 por padrão, limitada a 4, retries limitados e
lock por diretório para impedir dois processos de gastar simultaneamente no
mesmo trabalho. Cada tentativa reserva caracteres/custo/pedido **antes** de
chamar a API; falhas/retries também consomem o teto conservador. O ledger persiste
em `generation-ledger.json`. Não apagá-lo para retomar. O cache valida hash do
WAV e identidade de texto/provedor/voz/modelo/parâmetros. Reruns com cache válido
não chamam a API. Uma interrupção pode deixar uma reserva sem resultado: ela
continua contada; não há promessa de cobrança exatamente uma vez pelo provedor.

Resultados e parâmetros por exemplo ficam em `generation.jsonl` e no cache;
falhas em `generation-failures.json`. Todas as falhas permanecem inventariadas.
A montagem exige cache completo e íntegro, não pula frases que falharam.
Não aparar clipes para acomodá-los: o master preserva todos seus samples e
insere silêncio digital, inicialmente 2 s, entre falas e bookends de 2 s.
Nenhuma fala de marcador é adicionada.

A montagem produz **um** `master_raw.wav`, `manifest.json`, `manifest.csv` e hash.
O manifesto registra IDs/textos/ordem/splits, taxa, limites em samples/segundos,
duração, hashes de geração e separadores. Padrão: WAV PCM16 mono 24 kHz. O script
normaliza taxas de origem apenas na montagem; registra os arquivos originais.
Preflight verifica o tamanho RIFF de 4 GiB e recusa um arquivo que ultrapasse o
limite. Duas horas a 24 kHz/PCM16/mono ocupam aproximadamente 346 MB, sem risco
nesse plano inicial. Caso um futuro corpus exceda o limite, implementar RF64 ou
outro **único** contêiner compatível com a DAW antes de gerar; nunca transferir
uma divisão manual para o proprietário.

## A única operação manual

1. Abrir **somente `master_raw.wav`** no Melodyne/DAW.
2. Aplicar a receita aprovada a todo o material, em uma operação de trabalho.
3. Exportar **somente `master_processed.wav`**, WAV PCM16 mono, com a mesma taxa.

Não reordenar, remover ou duplicar trechos; não acelerar ou esticar globalmente.
Alterações de pitch/formantes podem manter a duração. Preservar a linha do tempo
e os separadores; exportar a faixa completa, incluindo silêncio nas extremidades.
A etapa não exige importar, editar, exportar ou cortar centenas de clipes.
O script não promete controlar o Melodyne nem substitui a receita por um efeito próprio.

## Recorte, conteúdo e quarentena

```bash
python3 scripts/dataset_pipeline.py extract \
  --manifest /CAMINHO_PRIVADO/pilot/manifest.json \
  --processed /CAMINHO_PRIVADO/pilot/master_processed.wav \
  --output-dir /CAMINHO_PRIVADO/pilot/processed
```

O manifesto define IDs e ordem. Se a duração coincide e o interior de todos os
separadores é validado, usa os limites conhecidos. Em exportações com padding ou
resampling, normaliza a taxa internamente e faz correspondência **monótona** de
separadores. Pausas internas longas são candidatos extras, não novos exemplos.
Cada clipe depende de suas duas âncoras; ausência/ambiguidade põe os vizinhos em
quarentena, sem deslocar as transcrições seguintes. Se faltar o primeiro ou último segmento em um layout repetitivo, durações/envelopes iguais não provam identidade: toda a região sem candidato único fica em quarentena. Em trechos com duração/envelope distintivos, landmarks únicos e não cruzados recuperam a correspondência sem cortar o ataque dos seguintes. Posições normalizadas nunca bastam para aprovar esse caso. As margens conservam 80 ms
antes e 120 ms depois da atividade detectada, incluindo pequenas caudas. Caudas
maiores, ruído que preencha separadores ou alterações extensas de tempo podem
exigir revisão local em vez de alinhamento automático.

`processed/dataset.jsonl` mantém a quantidade e ordem esperadas, inclusive
registros em quarentena. `clips/` contém resultados aceitos tecnicamente;
`quarantine/` contém afetados. `validation-report.json` registra âncoras,
padding/taxa, motivos e pendências. O schema versionado está em
`dataset/processed-example.schema.json`. São verificados hashes/proveniência,
segmentos vazios/fundidos/duplicados, duração, nível/clipping, envelope e alteração
global coerente de duração. O algoritmo é conservador; detecção por envelope
não prova palavras, pronúncia ou identidade perceptiva.

Todos os resultados começam com `content_review=pending`. Para checar conteúdo,
o adaptador opcional usa um **modelo ASR já existente em disco**, em CPU int8,
sem GPU nem download. Ambiente separado opcional:
`dataset/requirements-audit.txt`. O caminho local evita resolução automática de
checkpoints; a implementação é documentada pelo
[projeto faster-whisper](https://github.com/SYSTRAN/faster-whisper) e a versão
fixada existe no [PyPI 1.1.1](https://pypi.org/project/faster-whisper/1.1.1/).
O auditor não foi executado com fala real: falta áudio e modelo local autorizado.

```bash
python3 scripts/dataset_content_audit.py \
  --dataset /CAMINHO_PRIVADO/pilot/processed/dataset.jsonl \
  --model-dir /CAMINHO_LOCAL_ASR_EXISTENTE \
  --output-dir /CAMINHO_PRIVADO/pilot/content-audit
```

ASR usa `language=pt`, sem prompt com a transcrição esperada e sem VAD que possa
ocultar bordas. Preserva texto pedido e reconhecido, normaliza apenas a comparação
e sinaliza qualquer edição de palavras. Números/nomes podem gerar falsos positivos;
eles devem ser revisados, sem alterar silenciosamente áudio/transcrição. Fazer
revisão estratificada por categoria e split, incluindo primeiras consoantes,
números, nomes, frases longas, exemplos lentos e todos os sinalizados. Não é
necessário editar clipes manualmente para essa revisão.

A aprovação em lote é preparada por `scripts/dataset_review.py`: o proprietário
não escreve centenas de aprovações/evidências em JSON. Primeiro executar somente
o plano, sem gravação (padrão), sobre o dataset ainda pendente, o relatório ASR
e sua proveniência. `--corpus` deve apontar ao corpus exato aprovado.

```bash
python3 scripts/dataset_review.py \
  --dataset /CAMINHO_PRIVADO/pilot/processed/dataset.jsonl \
  --content-audit /CAMINHO_PRIVADO/pilot/content-audit/content-audit.jsonl \
  --audit-provenance /CAMINHO_PRIVADO/pilot/content-audit/content-audit-provenance.json \
  --corpus dataset/corpus.pt-BR.jsonl
```

O plano fornece `owner_review_template`, hashes exatos e `suggested_sampled_ids`,
com pelo menos um exemplo por estrato existente de categoria e split entre os
candidatos aprováveis. Salvar esse único template como `owner-review.json` privado.
Ouvir/conferir a amostra sugerida (ou outra com a mesma cobertura), incluindo
identidade/efeito, consoantes iniciais, números, nomes, prosódia e cortes. Preencher
apenas revisor/data/referência da aprovação, os IDs efetivamente conferidos e
`batch_sample_approved=true`, `perceptual_sample_approved=true` **depois** da
aprovação explícita do proprietário. O script não marca que alguém ouviu por conta
própria, e aprovação ASR sozinha não aprova qualidade perceptiva.

O template vincula SHA256 de corpus, dataset, relatório e proveniência ASR,
manifesto e masters bruto/tratado. O helper confere também o arquivo tratado real,
os hashes de todos os clipes, texto/grupo/split contra o corpus e recalcula a
comparação ASR. Alterar qualquer entrada exige novo plano e nova aprovação.
A categoria vem do corpus vinculado; ela não é inferida da transcrição reconhecida.

Somente exceções ASR precisam de decisão específica, em
`flagged_reviews: {"ID": {"decision": "approved", "evidence": "checagem feita"}}`
ou `decision: "rejected"`. Sem decisão, ficam `pending`, sem bloquear os bons.
Se uma exceção aprovada criar outro estrato, incluir também esse ID na amostra
conferida. Quarentena técnica e fixtures sintéticas nunca são promovidas pelo
helper. Não corrigir transcrições ou áudio para obter uma aprovação automática.

Para gerar a revisão em lote, repetir os argumentos do plano e acrescentar:

```text
--owner-review /CAMINHO_PRIVADO/owner-review.json \
--write-review --output /CAMINHO_PRIVADO/review.json
```

`--write-review` é opt-in; sem ele nada é gravado, mesmo com configuração aprovada.
A saída contém automaticamente a evidência e os hashes por exemplo. O antigo
`dataset/review.example.json` descreve o formato de saída individual e pode ser
usado para casos pequenos; não é uma obrigação de edição manual por clipe.
Reexecutar `extract --review /CAMINHO_PRIVADO/review.json`. O extrator confere
hash do export, manifesto e hashes de áudio vinculados à aprovação em lote.
Preservar o dataset/relatório ASR originais usados na revisão para rastreabilidade.
Rejeitados vão para quarentena; pendentes continuam fora do treino. O treinamento
só importa `quality_status=accepted`, `alignment_status=aligned`,
`content_review=approved`. A integração está em `training/`; nenhum treino pode
usar a demonstração de tons. Nenhuma revisão real foi produzida nesta entrega.

## Reprodução dos testes pequenos

```bash
python3 -m pytest -q tests/test_dataset_pipeline.py tests/test_dataset_review.py
python3 scripts/dataset_demo.py \
  --output-dir /var/tmp/aegis-tts-dataset-20261101/synthetic-demo
```

Trinta testes passaram: cache/hash, dry-run sem provedor, teto, aprovação vinculada
ao corpus, manifesto/WAV único, pausa interna de 3,2 s, separador alterado,
padding inicial/final, resampling, caudas, segmento ausente sem deslocar IDs,
stretch global, revisão vinculada ao export, comparação de conteúdo, WAV truncado, extremos ausentes em layout repetitivo, tetos NaN/Inf, preflight de memória, ordem alterada, deslocamento local com duração total preservada, clipping e duplicações.
Outros 17 testes sintéticos validam o helper de revisão: plano sem escrita, aprovação em lote sem decisões manuais por ID bom, cobertura por categoria/split, autorização perceptiva explícita, bindings de todos os hashes, flags pendentes/rejeitadas, recusa de auditoria adulterada, clipes alterados, fixtures e permissões privadas. Total: 47 testes.
Esses testes validam mecânica do pipeline; não validam fala PT-BR nem a receita
Melodyne. O próximo passo real continua sendo revisão/aprovação de corpus,
provedor/voz/direitos/orçamento, seguido do piloto autorizado e do único export tratado.
