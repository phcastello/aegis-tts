# Importação futura para F5-TTS e MagpieTTS

O importador `training.prepare_dataset` agora aceita `--architecture f5` e
`--architecture magpie` para a coleção `aegis.audio_collection.v1`.
É preparação técnica em CPU, sem treinamento. A receita `training/recipe.json`
continua sendo um registro histórico do piloto Piper descartado; não deve ser
usada como receita F5 ou Magpie. O padrão legado do importador permanece para
reproduzir evidências anteriores.

```bash
python3 -m training.prepare_dataset \
  --input-schema aegis.audio_collection.v1 --architecture f5 \
  --manifest /PRIVADO/novo-corpus/dataset.jsonl \
  --dataset-root /PRIVADO/novo-corpus \
  --output /PRIVADO/prepared/novo-corpus-f5-v1

python3 -m training.prepare_dataset \
  --input-schema aegis.audio_collection.v1 --architecture magpie \
  --manifest /PRIVADO/novo-corpus/dataset.jsonl \
  --dataset-root /PRIVADO/novo-corpus \
  --output /PRIVADO/prepared/novo-corpus-magpie-v1
```

Os comandos são exemplos para quando o novo corpus estiver disponível. Não foram
executados sobre um corpus privado nesta investigação. Use os WAVs tratados
originais, evitando converter o snapshot Piper de 22.050 Hz para F5 de 24.000 Hz.
Isso não recuperaria a banda já removida.

A implementação reaproveita as verificações existentes de RIFF/PCM16/PCM24,
hashes, duração, clipping, silêncio, IDs, grupos, áudio idêntico e variantes
textuais entre partições. Converte por `resample_poly` Kaiser beta 10 para
24.000 Hz no F5 e 22.050 Hz no Magpie, com PCM16 mono; recusa overshoot em vez
de limitar ou normalizar. Não altera pitch, efeitos, loudness, velocidade ou
margens. Os WAVs originais permanecem intactos. A taxa Magpie foi confirmada na
configuração do NanoCodec utilizado pelo modelo v2607; os detalhes estão em
[Magpie](magpie-evidence.md).

`dataset.jsonl` conserva o registro original completo em `source_record`, a
transcrição original, caminho/hash do WAV de origem, taxa/largura originais,
ID e grupo. `text` recebe somente NFC e remoção de espaços nas extremidades;
números, siglas e acentuação não são reescritos. O resultado inclui os hashes do
manifesto original, dos WAVs convertidos e das partições. Um plano opcional
`--split-plan` permanece vinculado ao manifesto e só move grupos inteiros de
treino para validação, preservando teste e validação originais.

Os manifests `f5-{train,validation,test}.jsonl` ou
`magpie-{train,validation,test}.jsonl` são **intermediários** com
`audio_filepath`, `text`, `duration`, ID e grupo; o teste fica separado do treino.
Não são anunciados como entradas finais prontas para o trainer:

- F5: executar o preparo oficial da revisão fixada para `raw.arrow`,
  `duration.json` e vocabulário; conferir cobertura do `vocab.txt` compatível com
  o checkpoint brasileiro antes de alterar seus embeddings.
- Magpie: fornecer `context_audio_filepath` e `context_text` de outro trecho do
  mesmo locutor e da mesma partição. A escolha não é inferida pelo importador.
  Usar `pt-BR` e o tokenizer `portuguese_Brazilian_phoneme` do modelo fixado no
  `dataset_meta`; o campo `speaker=0` é um rascunho de locutor único a revisar.
  Preparar códigos pelo NanoCodec oficial somente depois de validar o roundtrip.

Cada snapshot nasce com `content_review=pending`, `training_authorized=false` e
um `training-rights.pending.json` vinculado aos hashes do manifesto/dataset.
A importação não libera o treinamento. O template exige identificar o provedor,
URL/data dos termos e evidência dos direitos de voz e de uso dos áudios em
fine-tuning. Para outputs ElevenLabs, confirmar expressamente que os termos
aplicáveis permitem esse uso específico antes de qualquer piloto, incluindo uma
revisão separada se houver adaptação do vocoder/codec. Uma aprovação histórica de
outro corpus ou outra arquitetura não é automaticamente transferida.

Antes de um piloto: revisão auditiva/ASR, correspondência exata de transcrições,
roundtrip aprovado do caminho acústico escolhido, autorização do proprietário,
janela exclusiva de GPU e teste real de memória por batch. Não há runner de
fine-tuning F5/Magpie nesta alteração; os campos de aprovação registram evidência
futura e não substituem essas verificações.

O snapshot usa destino novo, staging e publicação imutável, com diretórios
privados e arquivos somente leitura. Não sobrescreve exportações existentes.
Os testes sintéticos verificam ambas as taxas, preservação dos originais e textos,
splits/IDs/grupos, rejeição de vazamento e ausência de autorização implícita.

Na inspeção atual, o caminho privado da coleção histórica indicado na documentação
não existe neste host. A referência protegida com transcrição exata tem 32 kHz,
não 48 kHz. Os arquivos de demonstração em `synthetic-demo` são tons rejeitados
para treinamento. Não foi fabricado um original de 48 kHz por upsampling; para
avaliar especificamente o novo material tratado em 48 kHz ainda é necessário que
ele esteja disponível, sem depender disso para a triagem da arquitetura.
