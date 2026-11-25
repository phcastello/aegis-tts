# Léxico de pronúncia F5 v2 — 25/11/2026

## Evidência e decisão

O gateway F5 ativo (`AegisVoicev1.0-F5-H-CONTINUOUS`, versão instalada
`20261113-h-stream-pronunciation`) já utiliza o normalizador de texto pt-BR
do Qwen. O processo define `AEGIS_F5_LEXICON_PATH` como
`~/.local/share/aegis-tts-f5/current/config/pt-BR.yaml`. O arquivo ativo
era idêntico, inclusive no SHA-256, ao léxico Qwen preservado (v1, hash
`7c67f461674d2a4eceb99278707245196dab7b543943a94d0a65aa9de704cd1f`).
Não havia perda geral de regras na troca de sintetizador. A substituição
histórica de Aegis era `Égide`, que não corresponde à pronúncia **Aéjis**
(/aˈɛʒis/) confirmada no
[levantamento de wakeword pelo proprietário](https://github.com/phcastello/Aegis/blob/55e8c3c9bc96897e3c7fad98806dc6a290fff1bf/docs/v0.8.0-wakeword-sherpa-poc.md#pronúncia-e-arquivo-de-keywords).
Gmail não tinha uma regra no léxico v1. Foram preservadas as demais
pronúncias da configuração anterior, respeitando seus limites de palavra e
sensibilidade a maiúsculas:

| Termo original | Texto exclusivo da síntese |
| --- | --- |
| Aegis (sem distinção de caixa) | Aéjis |
| Gmail (sem distinção de caixa) | gê mail |
| VPN (maiúsculas) | vê pê ene |
| DNS (maiúsculas) | dê ene ésse |
| GPU (maiúsculas) | gê pê u |
| Qwen (sem distinção de caixa) | Cuén |

O [léxico F5 v2](../config/pronunciation/pt-BR-f5-v2.yaml) é uma configuração
versionada do F5, não uma alteração na voz, nos pesos, no vocabulário do
modelo, no wakeword ou na mensagem mostrada ao usuário. O léxico e corpus
v1 continuam congelados para o Qwen e para o perfil de recuperação. O F5
já usa `SpeechTextNormalizer` centralmente, ao receber a solicitação, antes
da síntese/planejamento dos segmentos. Os endpoints nativo (`/v1/aegis/speech`)
e compatível (`/v1/audio/speech`) passam pelo mesmo caminho. Não introduzir
uma segunda transformação no backend ou nos clientes.

## Verificação isolada

```sh
pytest -q tests/test_f5_pronunciation_lexicon.py tests/test_aegis_speech_domain.py
PYTHONPATH=gateway python -m aegis_speech validate-profile
PYTHONPATH=gateway python -m aegis_speech run-pronunciation-corpus
python scripts/verify-f5-pronunciation-contract.py --runtime-dir /path/to/f5/runtime
```

O último comando importa uma cópia conferida de `runtime/app.py` do F5,
substituindo somente o motor de síntese por uma implementação CPU fictícia;
não inicia o modelo ou o serviço real. Verifica PCM em streaming, WAV manual,
as regras recebidas pelo sintetizador, o formato do áudio e a preservação
do texto original. A amostra sintética enviada ao F5 existente produziu
4,085 s (texto original) e 4,501 s (grafia fonética proposta) de áudio mono
PCM 24 kHz, na mesma voz F5 contínua, sem alteração de configuração ou
inferência simultânea. Não houve treinamento. Esses resultados não provam
a pronúncia acústica: a aceitação final é auditiva.

## Instalação do léxico

O F5 carrega o léxico uma única vez na inicialização. Para instalar, preparar
uma **nova versão independente** abaixo de `~/.local/share/aegis-tts-f5/versions/`
baseada exatamente na versão F5 ativa. Preservar todos os assets, configuração
de síntese, ambiente, manifesto e runtime; há versões anteriores que já
compartilham arquivos imutáveis por hardlink. Não usar symlink para assets
fora da raiz da versão: o validador exige caminhos relativos internos. Se a
versão candidata for criada preservando hardlinks, substituir atomicamente
apenas o `config/pt-BR.yaml` **da candidata** por um arquivo novo (nunca
editar esse hardlink in place). O arquivo de origem é o léxico F5 v2 deste
repositório. Verificar o SHA-256 do novo léxico, a integridade dos assets e
runtime e o teste do contrato HTTP acima antes de selecionar a versão.

O mecanismo existente `activate_version.py` realiza a troca reversível da
versão e tem um watchdog de rollback. Quando a instalação fizer parte da
tarefa solicitada, usar esse mecanismo com os caminhos conferidos, mantendo
`current` e as versões anteriores intactos até o rollout. Durante o período de validação, confirmar
`/ready`, o perfil original de voz e uma requisição curta de PCM/WAV; o teste
objetivo e a escuta dos arquivos de comparação sustentam a aceitação. Não
é necessário fazer deploy de API ou dos apps para uma troca exclusiva do
léxico no serviço F5.


### Aplicação verificada em 25/11/2026

Após a aprovação auditiva explícita do proprietário para a amostra proposta, o
servidor47 recebeu `20261125-h-stream-lexicon-v2` como nova versão, com a versão
`20261113-h-stream-pronunciation` preservada. Foi criado um diretório candidato
por hardlinks de arquivos imutáveis; somente o léxico recebeu inode novo. O
SHA-256 do léxico v2 é `5f7988869142ce657dadae1c4768e826b633594e288984f2d1147166b9a48283`.
Antes da troca, os seis assets e os 18 hashes do runtime foram validados
contra os manifestos, e o vocabulário foi conferido com o normalizador em
produção. A troca usou `activate_version.py activate` com watchdog independente
e rollback. Após readiness HTTP 200, perfil `AegisVoicev1.0-F5-H-CONTINUOUS`
inalterado, 99.840 bytes PCM mono a 24 kHz no endpoint nativo e WAV manual
mono a 24 kHz (49.920 frames) com a frase sintética `Aegis, abra o Gmail.`,
a versão foi confirmada via `activate_version.py commit`. Nenhum peso/modelo
foi alterado, nenhum arquivo da versão anterior foi removido, e não houve
nova infra paga. A aprovação acústica foi da amostra proposta; a validação
no aplicativo permanece disponível ao proprietário.
