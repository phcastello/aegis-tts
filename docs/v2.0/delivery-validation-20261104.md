# Validação das interfaces de entrega — 4 de novembro

Resultados de validação na RTX A2000 de 12 GB. Os sete textos comuns foram
executados três vezes, com concorrência 1, saída 24 kHz mono PCM16 e as mesmas
regras de silêncio inicial/FIFO do benchmark. Não houve avaliação do final6.

| Sistema e caminho medido | Válidas | Primeiro PCM p50 | Primeira fala p50 | RTF longo |
| --- | ---: | ---: | ---: | ---: |
| F5 especializado, NFE16, arquivos por segmento | 21/21 | 1797,62 ms | 2534,31 ms | 0,2510 |
| VoxCPM2, LoRA2800, gateway HTTP PCM, piloto histórico | 21/21 | 566,905 ms | 706,905 ms | 0,69246 |

A auditoria posterior encontrou os seis masters físicos nos conjuntos de
treino e validação históricos. Assim, esse Vox é um **piloto com sobreposição
de origem**, inelegível para a seleção estrita. Seus resultados medem essa
execução, sem comprovar generalização da identidade a outro master. Os novos
modelos de voz partirão de pesos base ou do FastPitch público, com nova divisão
por master. O F5 especializado continua como referência histórica preservada;
seus pesos e sua receita não foram substituídos.

A divisão prospectiva deixa em quarentena os 25 registros dos cinco grupos
que ligavam masters, preservando os originais. Restam 675 falas para treino
(2,346 h), 180 para validação (0,691 h) e 200 para teste (0,690 h), sem
colisões documentais de master, grupo, texto normalizado ou hashes de áudio.
A escolha usa duração e uma regra determinística, sem qualidade do teste.
Esse teste será reservado aos novos modelos iniciados de bases públicas:
seus exemplos já foram expostos aos modelos históricos. A independência
das gravações antes da montagem dos masters continua sem comprovação.

Cada processo recarregou os pesos salvos. Houve três aquecimentos externos da
mesma frase, sem aquecimento por caso, e seed fixo 40733133. Os aquecimentos
internos de cada backend são registrados separadamente. Com n=3 por caso,
p95 não é publicado. Cabeçalhos e silêncio inicial não contam como fala.

O F5 entrega arquivos completos de segmentos depois da ODE e do Vocos. Seus
tempos incluem normalização, síntese, conversão PCM e fechamento dos arquivos;
não representam HTTP nem streaming dentro de um segmento. O checkpoint é o
especializado `b3faf799…`, com referência 0586 da partição histórica de treino.

Vox usa `ptbr-numbers-v1`, NFE10, KV1024, DSP passthrough, FIR 48→24 kHz e
buffer inicial de 320 ms. O adaptador 2800 é `b60e4d7e…`. O processo novo levou
163,330 s até prontidão, com cache compilado existente de 70,8 MB. A primeira
chamada externa teve TTFA de 758,951 ms; esse valor e o tempo de carregamento são
medidas separadas. Não se somam medições independentes para inventar TTFA.

Ambos tiveram zero lacunas pela regra de 10 ms e zero clipping. Vox também
passou WAV completo, PCM, status/SSE, cancelamento depois de 4096 bytes PCM,
nova chamada após cancelamento, saúde, prontidão, fila vazia e smoke. Seu
pico amostrado de VRAM foi 6086 MiB. Os servidores temporários terminaram e
a GPU ficou vazia antes da retomada do treino.

Uma tentativa anterior do novo executor Vox terminou antes dos aquecimentos
externos e da matriz: exigia `Voice-Profile` em `/ready`. O contrato real
fornece esse cabeçalho nas respostas de fala. A correção mantém a checagem
estrita de perfil na fala e verifica os campos reais de prontidão. A tentativa
abortada permanece registrada; não forneceu métricas de síntese.

## Clareza: duas referências comuns

ASR small local, revisão 536b066…, CPU int8, dois threads, scorer 4ee9f7da…,
mesmas versões e normalização fixadas. Foi analisada a repetição0 de cada
texto. Todos os 14 WAVs foram byte-idênticos aos históricos correspondentes;
as contagens abaixo foram reproduzidas em uma nova execução do reconhecedor.

| Sistema | Edições raw / 135 palavras | Edições spoken / 154 | Edições CER spoken / 838 |
| --- | ---: | ---: | ---: |
| Qwen, medição inicial de produção |21 |39 |109 |
| F5 especializado |26 |36 |97 |
| VoxCPM2 2800 |15 |34 |102 |

Vox melhora as contagens observadas do Qwen nas duas referências comuns.
Contra o F5, a vantagem de 11 edições raw cai para 2 spoken, e o CER spoken
favorece o F5. Números, siglas e grafia de nomes explicam parte da diferença.
As referências effective são específicas de cada frontend e têm denominadores
diferentes; seus percentuais não são uma comparação comum entre sistemas.

Não houve escuta. ASR, espectro e similaridade de locutor não aprovam a
identidade robótica nem comprovam acentuação ou omissões audíveis. Permanecem
questões concretas de nomes, siglas e conteúdo para avaliação auditiva.

A TTFA inicial do Qwen foi 1827,57 ms, com o perfil protegido
`AegisVoicev1.0-Q00-W04-P07-M02` e WORLD de produção. Aquela execução usou
a política anterior de aquecimento externo. Uma nova interface para igualar
essa política ainda requer qualificação real; a diferença temporal acima
não é a comparação final pareada com o Qwen.

## FastPitch: estágio de idioma

No controle CUDA de qualidade, BF16 acústico com parâmetros/vocoder FP32,
o checkpoint público 16k fez 25/135 edições raw e 45/154 spoken; o 10k fez 39/135
e 57/154. Os dois produziram 7/7 WAVs válidos com modelos inteiramente na GPU.
O 16k ainda excede a tolerância de 2 pp frente ao Qwen nas duas vistas comuns.
Esses pesos ainda não foram especializados com a voz da Aegis.

Em CPU FP32, os mesmos checkpoints fizeram 35/135 e 34/135. A diferença entre
CPU e GPU muda de sinal conforme o checkpoint; a ordem 10k→16k se mantém,
mas a magnitude da melhora muda. Não foi isolada uma causa entre dispositivo,
precisão e operações do backend.

O checkpoint exato 18k também produziu 7/7 WAVs válidos na GPU, mas fez
38/135 edições raw, 57/154 spoken e 147/838 edições de caracteres spoken.
Essas medidas pioraram frente ao 16k. O 18k não substituiu o melhor retido;
seu arquivo de pesos temporário foi removido depois da decisão, mantendo
amostras, métricas, hashes e os estados completos de treino e melhor modelo.

O treino retomou o estado completo 16500, incluindo Adam, scheduler, RNG e
posição nos dados; houve atualização positiva dos pesos depois da retomada.
A seleção final e a identidade continuam abertas. O Qwen permanece parado
na janela de pesquisa autorizada, com seu container e rollback preservados.
Depois, o treino foi pausado com estado completo no passo 18004 para medir
o checkpoint exato 18000 na GPU. Concluída a comparação, retomou o estado
18004 em um processo novo e voltou a atualizar pesos. A redução de taxa de
aprendizado prevista na receita ainda não ocorreu. A próxima decisão usa
validação, mantendo fechado o teste final.

Provas locais compactas:

- `operations/final-executor-preparation-20261104/f5-validation-v3-actual-receipt.json`.
- `operations/final-executor-preparation-20261104/vox-v4-readiness-contract-preparation/actual-validation-receipt.json` — SHA256 `e2f2c12d24e30f29014b3f9813794e2bfbe6a9f82dd7ce6b6a23b866e5ee68b7`.
- `operations/final-executor-preparation-20261104/delivery-comparison-asr-v3-v4/result.json` — SHA256 `ae90af757141c78f101d0d06c143d436c00337ba08a7261d3e0ea7f346cbfac1`.
- `asr/clarity-validation-audit-20261104/REPORT.private.md`.
- `operations/097d-fastpitch-actual-resume-proof-20261104.json`.
- `fastpitch/gpu-language-quality-preparation-20261104/asr-step18000/result.json` — SHA256 `5ed8d9f4b8e07a3724224e67fb5ca228e41e8b5e7e8b9e976a344fedd0a4d7f3`.
- `operations/097f-main-actual-resume-proof-20261104.json`.
- `data-audit/origin-master-split-v2-quarantine25-20261104/artifacts-final/split-recipe.private.json` — SHA256 `b9f5d6de13b2c65a211ffc25cfeb8a33447bb1cbe9cb800bc5006a5697efa7bd`.

Todos os caminhos relativos acima partem de `/var/tmp/aegis-v2-work`.
Áudios, transcrições privadas e pesos permanecem locais.
