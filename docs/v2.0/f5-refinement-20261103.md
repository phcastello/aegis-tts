# AegisVoice v2 — refinamento de inferência, 03/11/2026

Rodada concluída sem alterar pesos, arquitetura ou configuração do Qwen. O
`best`, época 29/update 3.741, continua preservado. A candidata técnica é
**referência 0586/NFE 8, léxico exclusivo, contexto Vocos e síntese por sentenças**.
Conteúdo e latência melhoraram neste ensaio; **robotização e pronúncia final
aguardam escuta**. Nenhum novo treinamento foi iniciado.

## Referências e pronúncia

Foram comparadas quatro falas completas do dataset final: `0586`, `0665`,
`0042` e `0015`, de grupos distintos do treino, com 10,10–11,90 s. Os hashes
original/preparado/mel e as transcrições conferem. A seleção cobre quartis
acústicos entre 143 referências elegíveis; não escolhe uma síntese excepcional.
O ASR das fontes sinaliza divergências em 0586/0042, mantidas para escuta,
sem reescrever as transcrições para concordar com o reconhecedor.

A matriz usa duas frases inéditas × quatro referências × NFE 8/16, mesma
seed e orçamento gerado por frase; duas finalistas recebem uma terceira frase.
O WER auxiliar favoreceu **0586/8: zero erros em 51 palavras**, contra 1/51
em 0586/16 e 1/51–2/51 em 0015/8–16. As demais referências foram avaliadas
em duas frases, sem comparação direta dos percentuais entre coortes diferentes.
Não houve ganho consistente de conteúdo ao trocar a referência. Distância
acústica foi somente desempate técnico, sem classificação de identidade.

O proprietário confirmou **A-é-jis**. O normalizador isolado
`training/f5_refine_text.py` preserva o texto exibido e compara oito clipes:
`A éjis`/`A-é-jis`, letras PT-BR/variantes anglicizadas, SUS como palavra/soletrado
e pares crus/expandidos de moeda, percentagem, data e horário. Defaults:
CPU `cê pê u`, API `a pê i`, CPF `cê pê efe`, SUS `sus`, URL `u erre ele`,
HTTPS `agá tê tê pê ésse`. Não há regra universal de soletração.

Data/horário expandidos foram recuperados pelo ASR; os crus divergiram.
Nas variantes do nome, reconheceu `Raezes` e `a Aegis`; a pronúncia não está
resolvida. CPU/API/URL e centavos exigem escuta. WER de grafias fonéticas,
siglas e números não é nota de pronúncia. O Qwen manteve seu frontend real,
inclusive `Aegis → Égide`; sua representação foi auditada no container.

## Segmentação e cauda

O recorte antigo deixava passar **um frame condicionado: 10,667 ms/256 samples**.
O novo decoder usa o tamanho real do mel e 32 frames autênticos de contexto
esquerdo para o Vocos, removendo somente esse prefixo conhecido. Em CPU,
o sufixo coincide exatamente com a decodificação completa; no A/B CUDA,
as diferenças entre decoder frio/contextual desaparecem após 400 ms.
O pacote inclui três decodificações do mesmo mel para conferir o início.
Seus tempos cumulativos incluem ASR e não são benchmark de desempenho.

A cauda longa é outro problema. O F5 gera duração fixa sem EOS semântico;
a razão de bytes não verifica se as palavras terminaram. O contexto testado
soma 33,291 s, contra máximo de 18,633 s nas falas de treino, mas seus
3.121 frames/449 tokens não excedem o limite efetivo de 8.192 posições.
Desvio de alinhamento/duração é hipótese, não causa única demonstrada.

Reduzir o orçamento neural para 90%, mantendo texto completo e decoder igual,
piorou o WER de **18,37% para 22,45%**. Foi rejeitado. A candidata segmentada
teve **WER zero no texto longo**, incluindo a última frase, contra
22,45%/20,41% nas versões originais NFE 16/8. Preserva texto, pontuação e todos
os samples gerados, sem VAD, fade, ganho, efeitos ou cortes de cauda/consoantes.

Fontes fixadas: [CFM/duração](https://github.com/SWivid/F5-TTS/blob/45868abaabd74aabf03cd673d6dd00cc9d9c01ae/src/f5_tts/model/cfm.py),
[DiT/contexto](https://github.com/SWivid/F5-TTS/blob/45868abaabd74aabf03cd673d6dd00cc9d9c01ae/src/f5_tts/model/backbones/dit.py).

## Comparação controlada

Três textos comuns: curta inédita, nome/siglas e três sentenças longas.
Uma observação por condição; medições completas em CSV/JSON.

| Condição | PCM útil curta, s | Siglas, s | Longa, s | RTF mediano |
| --- | ---: | ---: | ---: | ---: |
| Qwen atual | 1,685 | 1,639 | 1,664 | 0,663 |
| F5 original NFE 16 | 1,690 | 1,795 | 3,951 | 0,351 |
| F5 original NFE 8 | 0,827 | 0,885 | — | 0,173 |
| Candidata refinada | 0,856 | 0,851 | 0,988 | 0,214 |

Qwen inclui HTTP/DSP; F5 é inferência local, excluindo carga única. PCM útil
é proxy digital de duas janelas de 10 ms, não prova de inteligibilidade.
Não há streaming neural dentro do bloco. O FIFO instrumentado inclui gravação,
métricas e releitura, sem HTTP.

O NFE 8 original longo apresentou **25 samples FLOAT fora de [-1,1], pico
1,4471**. Sua conversão PCM16 foi recusada; o original permanece intacto.
A candidata segmentada não apresenta overshoot. Nos dez pares da matriz,
ambos os NFEs têm saídas finitas sem overshoot; WER agrupado 1,85%/2,47%,
flatness mediana 0,01725/0,01429 e extensão YIN p90–p10 112,1/81,4 Hz em
8/16. Isso não distingue textura robótica desejada de degradação acústica.
**NFE 8 não foi aprovado globalmente nem declarado superior.**

Na longa segmentada, blocos chegam em **0,988/1,961/3,073 s**, sem underrun
FIFO estimado; saltos nas junções 0,000157/0,000097. Outra seed na curta também
teve WER zero. Identidade, aspereza, consoantes e continuidade perceptiva
permanecem pendentes na ficha de escuta, sem notas automáticas.

## Decisão de treinamento

Todos os 363 tensores de parâmetros foram ajustados anteriormente
(337.096.804 parâmetros); Vocos/frontend mel ficaram fixos. O objetivo CFM
não supervisiona explicitamente identidade robótica. Prior humano e
representação acústica são hipóteses; o roundtrip Vocos aprovado impede
atribuir a causa ao vocoder por suposição. As losses disponíveis não
demonstram overfitting tardio nem comprovam aprendizagem da textura desejada.

Somente se a nova escuta continuar insatisfatória em várias frases:
identificar primeiro o atributo ausente e considerar piloto separado de
200 updates, teto 400, LR 1e-6. Uma hipótese é adaptar entrada acústica,
últimos quatro blocos DiT e saída (~65,77 M parâmetros), congelando encoder
textual/Vocos. Avaliar a cada 50 updates com treino representativo, 55 falas
de validação para regressão e frases novas para A/B; não otimizar nas quatro
referências nem no teste já observado. Parar sem ganho consistente ou com
perda de conteúdo/fonética/continuidade. **Não executar automaticamente.**

## Entrega e preservação

Raiz privada: `/var/tmp/aegisvoice-v2-refinement-20261103/`.
**Pacote recomendado:** `aegisvoice-v2-refinement-compact-20261103.zip`,
**17.699.493 bytes**. Extrair e abrir `index.html`, também disponível em
`listening-compact/index.html`. São **12 comparações principais**, com ensaios
de apoio recolhidos: 49 clipes distintos, um WAV por clipe, sem formatos
repetidos. CRC, 54 links e todos os hashes passaram. Os 49 nativos e o ZIP
completo de auditoria (49.346.814 bytes) permanecem intactos.

SHA-256 do compacto:
`3014a42e9293d7a22cc7afd9adba2e5e9320a2dd288267010fe71e68638834b1`.

`selected-config.json` fixa parâmetros/léxico/orçamento real;
`technical-audit.json` sela os cinco checkpoints/estado retomável e contém
pares NFE, descritores, segunda seed e duração. `report.json`, métricas e ficha
acompanham a audição. Reempacotamento CPU: módulos `training.f5_refine_audit`,
`training.f5_refine_report`, `training.f5_refine_compact`, no ambiente F5 registrado.
Novo ensaio CUDA exige nova raiz e guardião `training.av2_ops`.

Qwen restaurado às **03:53:43 UTC**, saudável, com WAV/PCM reais, mesmo
container `08004c66f014`, imagem/configuração e 77 hashes protegidos; VRAM
10.674 MiB. Janela total: **17min06,799s**. Todos os checkpoints mantêm seus
hashes. Não houve promoção à produção; as alterações preexistentes do workspace
foram preservadas.

**77 testes CPU**, compilação e `git diff --check` aprovados. Commits:
`1737a86`, `395d747`, `56d3863`, `8e778c9`, `845de60`, em Conventional Commits.
Áudios, dados e pesos ficaram fora do Git.
