# Piper: texto, fonemas e proposta para a etapa longa

O lowercase herdado do preparo F5 não é neutro para o eSpeak do Piper.
Comparado com `original_text`, muda os IDs de 915/1025 falas de treino e 51/55
de validação. A proposta para a etapa longa conserva a caixa e as palavras do
rótulo original, sem aplicar globalmente o refinement F5. O piloto continua
congelado; nenhum metadata ativo, áudio, cache, peso ou fila GPU foi alterado.

O audit só leu os 1080 registros treino/validação; não abriu os 50 de teste.
Executou contagem e G2P isolado em CPU, nice 19/uma thread: 5,12 s wall e 3,98 s CPU,
sem Torch, modelo, ASR ou leitura/regravação de áudio. Artefatos privados em
`/var/tmp/aegis-v2-work/piper/text-phone-audit/`.

## Cobertura textual

Todos os 1080 `text` ativos são exatamente `original_text.lower()`. Não há
dígitos, 14h05, datas numéricas ou siglas em maiúsculas. A busca adicional por
CPU/GPU/API/CPF/CNPJ/SUS/URL/HTTPS/HTTP/XML/JSON/PDF/USB/SSD/RAM/CEP/RG sem
distinguir caixa também não encontrou ocorrências. Isso não é uma busca
semântica universal de todas as possíveis abreviações.

Os numerais presentes já estão escritos por extenso. As contagens abaixo são
candidatos lexicais e se sobrepõem; não são uma anotação semântica do áudio.
`um/uma/uns/umas` ficam separados porque também são artigos.

| Categoria | Treino: falas/ocorrências | Validação: falas/ocorrências |
| --- | ---: | ---: |
| Aegis | 2/2 | 0/0 |
| Dígitos, horário numérico, data numérica | 0/0 | 0/0 |
| Siglas maiúsculas ou léxico técnico consultado | 0/0 | 0/0 |
| Cardinais não ambíguos, excluindo um/uma | 162/273 | 13/19 |
| um/uma/uns/umas, ambíguos | 602/888 | 34/44 |
| Ordinais | 112/149 | 6/8 |
| Unidades hora/minuto/segundo | 49/67 | 6/6 |
| Candidato estrito a horário por extenso | 4/5 | 1/1 |
| Meses | 7/13 | 0/0 |
| Dias da semana | 7/9 | 0/0 |
| Candidato a data por extenso, número+de+mês | 3/4 | 0/0 |
| Falas reescritas pelo f5-refinement | 2 | 0 |

As únicas regras F5 aplicadas são duas substituições `lexicon:Aegis`.
Nenhum horário, número ou sigla do treino seria expandido pelo refinement.
Portanto, o problema observado com14h05 nas frases de inferência é uma lacuna
de interpretação do formato de entrada, não evidência de que os rótulos do
treino contenham esse formato mal falado.

## Efeito de caixa e fronteiras

Fresh lowercase reproduziu todos os966 caches divergentes; não se encontrou
cache corrompido nesses casos. O eSpeak deixa de reconhecer muitos pontos como
fim de sentença quando a palavra seguinte foi convertida para minúscula.
Além de retirar pontuação/fronteiras, altera ligações e formas fonéticas.

| Medida | Treino lower→original | Validação lower→original |
| --- | ---: | ---: |
| Sentenças/BOS reconhecidos | 1065→1986 | 59→110 |
| Pontos conservados nos IDs | 1005→1926 | 53→104 |
| Sequências de IDs diferentes | 915 | 51 |
| Diferença persiste sem espaços/pontuação/stress | 370 | 13 |
| U+0303 no proposto | 9787 | 520 |
| Falas com U+0303 no proposto | 1015 | 55 |

Nos 966 casos divergentes, são 972 pontos e 972 fronteiras a mais no original.
Em 383 falas, a diferença não se resume aos marcadores: aparecem, por exemplo,
ligações `/s/→/z/` e mudanças de vogal/diftongo na versão conectada. São
divergências de frontend observadas; sem audição/alignamento não se rotulam
todas como erros fonéticos do áudio.

A agregação foi conferida na fonte do runtime pinado,
`piper/train/vits/dataset.py` SHA
`287c95109744a7234cc9508fb446ab281f9abe84a7ff3af07652d236cb35387b`:
o treino concatena `phonemes_to_ids(sentence)` **para cada sentença**, com
BOS/PAD/EOS por sentença. Não achata todos os fonemas para usar um único par
externo. O helper atual e a proposta seguem essa agregação. Um teste isolado
de duas sentenças reproduziu exatamente o proposto e demonstrou que a outra
agregação daria IDs distintos; recibo `aggregation-proof.json`.

## Voz de origem e o nome

O histórico do pacote descreve cortes dos masters tratados e aceleração
RubberBand1,0833886494, pitch1 e formantes preservados. Textos, grupos e
partições foram conservados. Não fornece dicionário de pronúncia nem captura
da representação fonêmica usada pelo sintetizador que gerou os masters.
Assim, nem o texto escrito nem o G2P atual demonstram os phones do áudio.

As três âncoras já transcritas (`0006`,`0007`,`0008`, todas validação) têm hashes
atuais correspondentes e zero edits no ASR fixo, mas não contêm Aegis, dígitos,
horários formatados ou siglas. São evidência limitada de correspondência em
fala comum; não resolvem o nome. Não foi feito novo ASR ou audição.

O proprietário confirmou a intenção A-é-jis. O eSpeak atual produz estas
representações isoladas, sem sintetizar áudio:

| Texto de entrada | Fonemas |
| --- | --- |
| Aegis/aegis | `ˌaeʒˈis` |
| A éjis | `a ˈɛʒis` |
| A-é-jis literal | `aɛʒˈis` |
| Aégis/Aéjis | `ˌaˈɛʒis` |

Copiar os hífens da representação desejada não fixa o stress no eSpeak.
As duas falas de treino com o nome (`0911`,`0960`) exigem decisão localizada
de GT antes de mudar seus rótulos fonêmicos. Não se aplicou nova pronúncia ao
áudio por suposição, nem se confundiu o ASR de uma voz Cadu não adaptada com
prova sobre a gravação da Aegis. Um fallback possível é quarentenar esses dois
casos na receita longa; isso não foi escolhido ou aplicado nesta proposta.

## Metadata proposto e próximo ponto de decisão

Arquivo novo, somente leitura e **não selecionado/não enfileirado**:

`/var/tmp/aegis-v2-work/piper/text-phone-audit/proposal-originalcase-native161-v1/target-originalcase-native161.PROPOSED.json`

SHA256 `b212a8eb20fe766af8d20fe741a20a07450022abd25a5bf20bf2ed84da7284a9`;
fingerprint `59568c022a7339165768322a456488eb2cd93bc2e3358f0e93dda99d07b6a99f`.
Os JSONLs derivados estão no subdiretório `prepared/`; só o texto passou para
`original_text`, e os IDs fonêmicos foram derivados dessa forma. IDs, grupos,
partições, durações, caminhos e hashes de áudio ficaram iguais; os sete probes
também ficaram iguais. O mapa original 161 e a nasalização U+0303/ID141 foram
preservados, sem OOV. Não se recalcularam WAVs, mels ou caches acústicos.

`derivation-receipt.json`, `case-boundary-diffs.private.json`, `summary.json` e
`per-record.private.json` conservam fontes/hashes, contagens, diffs e os dois
casos do nome. A proposta mantém 1025/55, conforme solicitado. A validação tem
os phones`r`/ID30 e`ə`/ID59 ausentes do treino; já existem no mapa da base.
Isso é oportunidade de medir generalização, não motivo para retirar validação.

Recomendação: preservar o piloto congelado como prova de engenharia, escolher
explicitamente uma nova receita original-case para a etapa longa e resolver
a política dos dois nomes separadamente. Não retomar silenciosamente o
checkpoint do piloto com outro fingerprint. O frontend de inferência para
números/datas deve ser um controle separado e igual antes/depois do target;
não é justificativa para reescrever todos os rótulos GT com o léxico F5.
Pronúncia, identidade e generalização continuam exigindo as saídas reais e os
testes comuns; cobertura de símbolos não os aprova.

## Limite da revisão de treinamento

O GPU084 revelou uma lacuna não coberta pela revisão estática anterior nem
pelos mocks: validação sob `inference_mode` criou caches persistentes STFT que
falharam no backward. O085 foi0→4 desde o checkpoint inicial, não3→4. Isso
continua sendo registrado como tal.

Revisão somente leitura do fix SHA
`2e08e891f89a766eaaaff985c5482cc63d528f199aa453ff9520f985eeb64911`
conferiu AST: só `AudioCache.get` e `validation` mudaram. Criação persistente
agora desativa inference_mode; validação usa no_grad. O autor registrou oito
testes CPU incluindo mel real/cache→backward e reprodução do erro original,
CUDA não inicializada. Nenhum desses fatos substitui o novo piloto real
3updates→checkpoint→processo novo→update4 na A2000.
