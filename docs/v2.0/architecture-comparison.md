# AegisVoice v2.0 — F5-TTS PT-BR e MagpieTTS

Estado em 02/11/2026: **manter Qwen 0.6B em produção; priorizar F5-TTS PT-BR
para o ensaio exclusivo na A2000**. Foram baixados pesos específicos, instalado
ambiente isolado, executados os dois caminhos de reconstrução acústica em CPU,
testados os frontends e concluída síntese F5 real em NFE 8/16/32. Não houve treinamento, parada
do TTS nem promoção de voz. Nenhuma candidata demonstrou ainda superioridade
perceptiva e operacional suficiente para substituir o baseline.

A seleção definitiva depende de escuta e medições CUDA. Isto é uma triagem
executada com limites explícitos, não uma alegação de que um modelo venceu por
licença, frequência nominal, número de parâmetros ou um score automático.

## Resultado por candidata

| Candidata | Evidência local desta investigação | Inferência / streaming | Treino em 12 GB | Decisão |
| --- | --- | --- | --- | --- |
| Qwen3-TTS 0.6B / vLLM-Omni | Serviço original saudável; nove WAVs novos da coorte fixa para escuta; baseline histórico preservado | 24 kHz; incremental no runtime implantado; histórico PCM útil mediano 1,653 s, reprodução estimada 1,756 s, EOF 2,588 s, RTF 0,779 | Receita acústica existente; codec congelado; memória de treino não medida | Baseline operacional mantido |
| firstpixel F5-TTS PT-BR | Safetensors de 1,348 GB com hash verificado; carga Base estrita/2545 tokens; Vocos real reconstruído em CPU; síntese inédita em NFE8/16/32 | 24 kHz; solução de frase completa antes de PCM; NFE 8/16/32 preparados; CUDA não medida | Hipótese plausível batch 1, checkpoint de ativações e acumulação; estados Adam+EMA ~6,28 GiB antes de ativações | **Primeira candidata para ensaio**; pesos CC BY-NC 4.0 |
| MagpieTTS v2607 / NanoCodec | Config e dicionário PT-BR inspecionados; G2P executado; codec oficial 425 MB verificado e roundtrip numérico executado em CPU | 22.050 Hz; síntese completa e streaming local não medidos; sem clonagem zero-shot | Novo locutor oficial; ~5,42 GiB só estados Adam FP32; codec congelado; runtime/embeddings v2607 ainda precisam de validação | Segunda candidata, condicionada à retenção do efeito e integração do frontend |
| VoxCPM2 | Preparação anterior reaproveitada; nenhum novo peso/runtime baixado | Decoder 48 kHz, encoder 16 kHz; streaming nativo disponível no código, A2000 não medida | Receita oficial estima ~20 GB para LoRA, acima do hardware | Exploratória; não iniciar outra investigação pesada antes de concluir F5 |

Números históricos Qwen vêm de nove chamadas aquecidas de 01/11, não de uma
comparação contra o smoke CPU F5. As novas amostras Qwen são para escuta e smoke
real do serviço, sem exclusividade e com F5 em CPU simultaneamente; seus tempos
não entram em ranking de arquiteturas. Não extrapolar CPU para GPU.

Fontes, pins, licenças e comandos: [F5](f5-evidence.md),
[Magpie/NanoCodec](magpie-evidence.md), [Vox/Qwen](voxcpm-qwen-evidence.md),
[baseline recalculado](qwen-historical-baseline.json) e
[benchmark de produção](../v0.8.0/production-latency-20261101.md).

Piper permanece descartado. Qwen 1.7B acelerado não foi repetido.
GPT-SoVITS PT-BR permanece suspenso pela falta do frontend original;
nenhum símbolo foi inventado. **Taxa abaixo de 32 kHz não elimina candidata**.
As prioridades e o antigo requisito de frequência dos relatórios anteriores
foram substituídos por esta orientação do proprietário.

## Reconstrução acústica efetivamente executada

Usou-se o trecho robótico já selecionado da v1.0, WORLD W04/P07/M02,
5,04 s, mono FLOAT **24 kHz**, com texto exato do manifesto de origem.
A referência natural protegida de 29,35 s tem 32 kHz. O caminho privado do corpus
histórico está ausente; nenhum original tratado de 48 kHz foi localizado.
Não se fabricou esse original por upsampling, não se cortou referência com
transcrição incompatível e nenhum original foi sobrescrito.

| Caminho | A → B → C | Resultado técnico CPU, dois threads |
| --- | --- | --- |
| F5 / Vocos | Original tratado24k → entrada FLOAT24k → F5 MelSpec(vocos) → Vocos.decode | Mel 8,40 ms; decode 361,40 ms; saída 5,0347 s; pico 0,423; nenhum valor fora da faixa |
| Magpie / NanoCodec | Mesmo original24k → FLOAT22050 → pad1024 → encoder/FSQ/decoder | Carga 7,50 s; encode 4,83 s; decode 6,50 s; roundtrip RTF 2,249; pad final 21,95 ms preservado; sem NaN/clipping |

Vocos foi carregado estritamente no código F5 fixado. O NanoCodec usou um
harness numérico de classes oficiais com hashes e carga estrita: remove metadados
NeMo/JIT e substitui a classe base por `nn.Module`, preservando as fórmulas.
**Não comprova o runtime NeMo completo.** Duração/picos/bandas não comprovam
inteligibilidade ou retenção da robotização. Ataques, sibilantes, ruído,
formantes e identidade precisam da escuta A/B.

O checkpoint brasileiro usa o caminho Vocos. BigVGAN possui outra definição de
mel e um checkpoint acústico upstream separado; não é uma substituição validada
só porque ambas as entradas têm 100 bandas. Se Vocos perder o efeito, há rota
para adaptação separada com pares waveform→mel fixo e decoder treinável,
condicionada a direitos e benchmark. Adaptar somente o F5 não corrige uma perda
sistemática do vocoder. No Magpie, a receita mantém NanoCodec congelado; perda
importante do efeito no roundtrip pesará contra sua escolha.

Smoke de síntese F5 CPU concluído, uma frase/seed, referência tratada em cache,
4,0853 s de áudio em todos os NFE: 8 → **210,741 s** (RTF51,585); 16 →
**428,067 s** (RTF104,781); 32 → **848,224 s** (RTF207,627). A difusão domina
o custo; vocoder ~0,28–0,33 s. Carga acústica 3,908 s; imports/carga 12,112 s.
Primeira chamada fria preservada, sem warm-up descartado; chamadas seguintes
aquecidas. Esses valores CPU não medem a A2000 nem provam a qualidade relativa
dos NFE. Carga estrita de 337.096.804 parâmetros confirmada.

## Frontends e pronúncia

F5 aceita os nove textos sem OOV, inclusive acentos, números e siglas. Isso
comprova cobertura do vocabulário, não normalização nem pronúncia. A inferência
inicial registra NFC/lowercase, sem reescrever números; o par `numbers_raw` /
`numbers_spoken` da coorte original separa esse problema. Não foi adotada a
substituição ingênua de cada sequência de dígitos do card, que não resolve
moeda/data/decimais corretamente. A auditoria textual do normalizador atual do gateway detectou moeda mal
expandida (`R$ 1.247,oitenta e três`) e soletração de SUS; não se deve importá-lo
como solução já correta para F5. Datas/horas/unidades foram expandidas. O léxico
atual mapeia Aegis para Égide, opção a comparar explicitamente com o nome cru.
Recibo privado `gateway-normalization-audit.json`; produção não foi alterada.

Magpie mostrou um risco concreto: o modelo seleciona tokenizer brasileiro com
`pt-BR`; o normalizador NeMo consultado aceita `pt`. Usar os argumentos do card
sem conferir o mapeamento pode selecionar inglês ou cair silenciosamente em
texto cru. O G2P isolado perdeu dígitos/símbolos em dois casos crus, enquanto a
versão numérica por extenso passou. A rota TN externo `pt` + tokenizer `pt-BR`
ainda precisa ser executada e ouvida. Não se atribuiu identidade à voz padrão.

A coorte original de nove textos permanece intacta e terá as mesmas repetições
por modelo/NFE. [Suplemento Aegis](../../benchmarks/architecture_compare/corpus.supplement.pt-BR.json)
identificado separadamente cobre o nome, que não estava nos nove textos.
Nenhuma pronúncia foi declarada correta apenas pelo tokenizer.

## Escuta privada e pacote para download

Raiz: `/var/tmp/aegis-v2-evaluation-20261102/` (0700; áudios/recibos 0600).
Foram preparados todos os pares A↔B, B↔C e A↔C dos dois caminhos, sem selecionar
melhores resultados. O índice usa nomes neutros e chave separada. Floats nativos
permanecem preservados; as cópias PCM16 para navegador têm quantização explícita,
sem ganho, resampling adicional, limiter ou trim. O ZIP solicitado pelo
proprietário foi produzido: `/home/phc/aegisvoice-v2-audicao-20261102.zip`,
**20.369.744 bytes**, 12 pares cegos, nove Qwen, três F5 e 12 transcrições ASR.
Contém HTML offline com exportação de notas JSON/CSV e chave separada em
`NAO_ABRIR_ANTES_DA_ESCUTA`. CRC, hashes e links locais foram conferidos.
SHA-256: `cd1ba4a1a9487e62436bfecb04eebd49fc0bf828759b3955027091deaf8f506c`.

Os nove WAVs Qwen novos usam exatamente a coorte fixa; o F5 CPU começa com
`novel_short` nos três NFE. Apenas textos iguais formam pares entre arquiteturas.
A redução CPU para uma frase é smoke técnico e material de escuta, não uma
avaliação suficiente de negativas, perguntas, siglas ou estabilidade longa.
Todos os nove casos/NFE serão executados na janela CUDA, conservando falhas.
Magpie tem reconstrução, mas ainda não síntese; Vox não possui nova amostra.

Registrar inteligibilidade, PT-BR, naturalidade, expressividade excessiva,
identidade, textura robótica, ataques/sibilantes, artefatos e preferência.
A escuta do proprietário permanece pendente. Um auditor ASR CPU local foi
baixado com revisão/hash fixados e transcreveu os nove Qwen e os F5 concluídos,
sem prompt, VAD ou envio externo. F5 NFE8/16/32 recuperou a frase integral; alertas
de moeda/siglas no Qwen são auxiliares e podem incluir falsos positivos.
Ver [ASR e limitações](asr-evidence.md). Nenhuma preferência foi inferida do ASR.

## GPU, produção e autorização operacional que falta

GPU observada: RTX A2000, **11.514 MiB visíveis**, driver **535.309.01**;
Qwen residente **10.674 MiB**. Não foi carregado modelo experimental em CUDA.
Branch de investigação: `feat/v2-f5-magpie-evaluation`, criada sobre
`0465472`; alterações locais anteriores, inclusive as sete remoções de dataset,
foram preservadas. Nenhum `AGENTS.md` localizado no checkout/pais.

Produção: container `aegis-tts`, imagem `aegis-tts-gateway:latency-1749f5f`,
ID `sha256:8aef74c67fe5238ce1e2b8781b9da3aedc73784014dd38bc1c1553f741fe93a0`.
Snapshot privado de inspect, `.env` e Compose, imagem retida e procedimento
`rollback-current/rollback.py` estão preparados. Para um ensaio que apenas pare
o container, o rollback inicia **o mesmo container**, sem reconstruir imagem,
trocar configuração ou apagar volumes. Confere ID/imagem, exige GPU livre antes
de restaurar, aguarda readiness e faz síntese real. Não foi necessário executá-lo.

**Proposta concreta: janela exclusiva de até 30 minutos para F5**, com início
UTC a combinar e últimos **8 minutos reservados à restauração**. Falta aprovação
explícita do intervalo e da breve indisponibilidade do TTS. O pedido atual de
preparação/download não autoriza essa interrupção. Não há impedimento de licença
para esta investigação não comercial com F5; uso comercial continua restringido.

Na janela: snapshot → parar somente TTS → conferir GPU vazia → carga fria e
pico → primeira chamada preservada/warm-up separado → coorte fixa NFE8/16/32
sequencial, referência em cache → suplemento Aegis → descarregar candidata →
restaurar container original → health/readiness/síntese real. Respeitar deadline;
se não houver tempo para todas as repetições, registrar incompletude e restaurar,
sem escolher resultados rápidos. OOM interrompe a rodada; nada de insistência
ou carregamento simultâneo com Qwen. O watchdog do probe não substitui rollback.

Métricas da rodada CUDA: carga fria, warm-up, VRAM residente/picos de carga e
inferência, normalização, conversão/mel da referência uma vez, custo de difusão,
vocoder, NFE/chamadas/blocos, primeiro PCM útil e reprodução estimada, EOF,
RTF, silêncio, caudas/cortes, falhas/OOM. O F5 entrega frase completa: ausência
de underrun com um bloco não é evidência de streaming neural.

## Dataset, orçamento e recomendação

O novo corpus menos expressivo não foi necessário para a triagem. Preparou-se
[importação F5/Magpie](dataset-import-v2.md) com formatos nativos, hashes,
transcrições originais, IDs/grupos/splits e revisão de direitos pendente.
`training/recipe.json` é histórico Piper e não foi reutilizado como receita.
Nenhum output ElevenLabs foi usado para treino. Fine-tuning completo, piloto
com dados privados e promoção da voz continuam exigindo as aprovações próprias.

A [conta do F5](f5-evidence.md) oferece rota plausível de treino na A2000,
mas nenhum forward/backward foi medido. Não existe estimativa real de duração
em horas: medir segundos/microstep, pico e custo de checkpoints após autorização,
e então projetar updates/acumulação/validação. Treinar modelo acústico e adaptar
vocoder são experimentos separados. Não se começa offloading enorme no Magpie
ou LoRA Vox acima do hardware apenas para contornar essa incerteza.

O preflight encontrou cerca de 48 GB livres, contra os ~16,65 GiB do relatório
antigo. O orçamento adicional é **no máximo 40 GB**, preservando margem de
operação. Ambiente, caches e modelos ficam separados da produção. O [inventário](preflight-f5-magpie-20261102.json) registra uso físico com
hardlinks contados uma vez, bytes dos downloads, versões, caminhos e espaço
restante; inclui o auditor ASR opcional de 486 MB. Uso físico final **9.48 GB**, incluindo ambientes/caches compartilhados e
ZIPs de verificação; **41.40 GB** livres no filesystem.
Nada preexistente foi apagado.

**Recomendação:** avançar F5 Base/Vocos como primeira candidata, sem o promover.
Se a escuta do roundtrip já reprovar os efeitos, resolver o vocoder antes de
investir no treino acústico. Magpie fica em segundo lugar, com atenção ao codec
congelado, frontend e adaptação real do locutor v2607. A decisão final só será
possível após audição e benchmark exclusivo; Qwen continua sendo a escolha
operacional demonstrada nesta A2000.

Validação de engenharia: **116 testes CPU passaram**, compilação Python, planos
sem GPU e integridade do ambiente passaram. A janela proposta é recusada pelos
guardas enquanto não aprovada. Isso não substitui os testes CUDA nem a escuta.
