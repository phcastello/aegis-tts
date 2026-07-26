# API de produção — Aegis TTS / vLLM-Omni

Base URL: `http://10.1.1.47:8001`

O serviço usa vLLM-Omni 0.22.0 e `Qwen/Qwen3-TTS-12Hz-0.6B-Base`. O endpoint
de síntese é compatível com OpenAI: `POST /v1/audio/speech`.

## Perfil de voz padrão

O gateway sempre usa `AegisVoicev1.0-Q00-W04-P07-M02`: a referência ICL
`Aninha_AegisVoice4.wav` e sua transcrição exata, ambas montadas somente para
leitura no diretório protegido. Não são devolvidas pela API, registradas nos
logs ou selecionáveis pelo cliente.

Os campos `ref_audio`, `ref_text` e `voice`, se enviados, são substituídos ou
removidos pelo gateway. Isso impede mudança pública de perfil. A referência
anterior `giselle_AegisVoice3.wav` não pertence ao runtime AegisVoicev1.0.

## Síntese simples com a voz padrão

```bash
curl --fail-with-body http://10.1.1.47:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Olá, Pedro. Esta é uma síntese usando a voz padrão.",
    "response_format": "wav"
  }' \
  --output resposta.wav
```

O servidor fixa Q00: `task_type: "Base"`, `language: "Portuguese"`,
`x_vector_only_mode: false`, temperatura 0,05, top-p 0,3, top-k 1,
penalty 1,15, seed 40733133 e `min_tokens: 0`. `non_streaming_mode: true`
preserva a montagem do prompt Q00, mesmo quando a resposta HTTP é streaming.
Toda resposta carrega `X-Aegis-Voice-Profile: AegisVoicev1.0-Q00-W04-P07-M02`.

## Referência própria

```bash
AUDIO_B64="$(base64 -w 0 minha-referencia.wav)"
REF_TEXT="A transcrição exata do áudio de referência."

curl --fail-with-body http://10.1.1.47:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d "$(jq -n \
    --arg input 'Olá. Esta voz usa a referência enviada nesta requisição.' \
    --arg ref_audio "data:audio/wav;base64,$AUDIO_B64" \
    --arg ref_text "$REF_TEXT" \
    '{input: $input, task_type: "Base", ref_audio: $ref_audio, ref_text: $ref_text,
      x_vector_only_mode: false, language: "Portuguese", response_format: "wav"}')" \
  --output resposta.wav
```

Esta forma de seleção não é suportada em produção: `ref_audio` e `ref_text`
são mantidos apenas por compatibilidade de esquema e são ignorados. Não envie
somente um speaker embedding.

## Streaming PCM

```bash
curl --no-buffer --fail-with-body http://10.1.1.47:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Teste de streaming com a voz padrão.",
    "stream": true,
    "stream_format": "audio",
    "response_format": "pcm"
  }' \
  --output resposta.pcm
```

O PCM retornado é mono, 16-bit, 24 kHz. Ele passa por segmentação acústica
WORLD W04/P07 e mastering M02 contínuo, independentemente do chunking HTTP.
Para analisá-lo como WAV:

```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i resposta.pcm resposta.wav
```

## Health e modelo carregado

```bash
curl --fail http://10.1.1.47:8001/health
curl --fail http://10.1.1.47:8001/v1/models
```

`/health` responde somente quando o backend vLLM-Omni estiver disponível;
`/health/ready` também exige hashes protegidos, DSP importável e warm-up interno
com a referência. O
modelo retornado em `/v1/models` é `/models/Qwen3-TTS-12Hz-0.6B-Base`.

## Limites e compatibilidade

- A rota antiga `/v1/synthesize` não faz parte do runtime padrão atual.
- O benchmark aprovado mediu RTF mediano em torno de 0,52; o desempenho de uma
  chamada individual varia com o texto e a carga da GPU.
- Não defina `min_tokens` positivo: ele atinge um bug conhecido na combinação
  Qwen3-TTS Base/vLLM-Omni 0.22.0 e pode causar erro CUDA.
- `AEGIS_VOICE_PROCESSING_MODE=streaming|offline|bypass` é somente operação
  administrativa. Para rollback para o serviço anterior, após parar esta
  composição execute:

```bash
docker compose -f docker-compose.eager-1.7b.yml up -d --build
```
## Contrato nativo Aegis v1

O gateway também oferece POST /v1/aegis/speech. Aceita somente
AegisVoicev1.0, prioridade 0 a 100 e as políticas enqueue,
replace_lower_priority, replace_all e reject_if_busy. A política não dá
significado ao texto: a aplicação chamadora decide a prioridade.
replace_all fica desabilitada até AEGIS_ALLOW_REPLACE_ALL=1.

Streaming retorna somente PCM mono s16le a 24 kHz. Eventos ficam separados em
GET /v1/aegis/speech/{request_id}/events como SSE e permitem reconexão por
Last-Event-ID. Estado e cancelamento usam GET/DELETE
/v1/aegis/speech/{request_id}; GET /v1/aegis/queue relata a fila.

Estados: accepted, normalizing, queued, generating, streaming, completed,
cancelling, cancelled e failed. A fila usa prioridade decrescente, ordem de
aceitação e índice de segmento; aging é min(10, floor(tempo_em_fila/30)).
Cliente desconectado ou parado cancela cooperativamente o upstream.

Operação:

    PYTHONPATH=gateway python -m aegis_speech validate-profile
    PYTHONPATH=gateway python -m aegis_speech validate-lexicon
    PYTHONPATH=gateway python -m aegis_speech normalize-text --text 'A GPU chegou a 62%.'
    PYTHONPATH=gateway python -m aegis_speech segment-text --text 'Primeira frase. Segunda frase.'
    PYTHONPATH=gateway python -m aegis_speech run-pronunciation-corpus
    PYTHONPATH=gateway python -m aegis_speech run-golden-regression
    pytest -q
    docker compose config

Saúde: /health é liveness; /ready e /health/ready exigem backend, hashes,
léxico e warm-up. Perfil, referência, Q00/W04/P07/M02, modelo e formato não
são controláveis pelo cliente. A porta é loopback por padrão; o profile é
montado somente leitura. Players Bluetooth devem manter o sink aberto: o corte
inicial conhecido é wake-up do sink, não do TTS.
