# API de produção

Base URL padrão: `http://127.0.0.1:8001`.

O gateway fixa o perfil `AegisVoicev1.0-Q00-W04-P07-M02`, Q00, WORLD W04,
P07, M02, seed, parâmetros de sampling, PCM s16le mono a 24 kHz e o pipeline
aprovado. Referência e transcrição são assets locais protegidos; clientes não
podem selecioná-las ou substituí-las.

## Saúde e observabilidade

- `GET /health`: liveness.
- `GET /health/ready` e `GET /ready`: backend e warm-up prontos.
- `GET /metrics`: métricas Prometheus de runtime sem texto de requisição.
- `GET /v1/aegis/queue`: estado da fila.

O healthcheck do Compose usa `/health/ready`; `/ready` é o alias público
documentado para automação simples.

## OpenAI-compatible

`POST /v1/audio/speech` aceita `input` (ou `text`), `stream` e
`response_format` (`wav` ou `pcm`). Respostas incluem os headers
`X-Aegis-Request-Id`, `X-Aegis-Voice-Profile`, `X-Aegis-Audio-Format`,
`X-Aegis-Sample-Rate` e `X-Aegis-Channels`.

```bash
curl --fail-with-body http://127.0.0.1:8001/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input":"Teste curto da Aegis.","response_format":"wav"}' \
  --output resposta.wav
```

## API Aegis

`POST /v1/aegis/speech` aceita:

```json
{"text":"Primeira frase. Segunda frase.","priority":50,"interrupt_policy":"enqueue","stream":true,"response_format":"pcm"}
```

Os contratos também incluem `GET /v1/aegis/speech/{request_id}`, `DELETE
/v1/aegis/speech/{request_id}` e `GET
/v1/aegis/speech/{request_id}/events`. Eventos SSE preservam os nomes e o
formato atuais. A prioridade é 0–100; as políticas aceitas são `enqueue`,
`replace_lower_priority`, `replace_all` e `reject_if_busy`.

Para WAV não streaming, envie `"stream":false,"response_format":"wav"`.
Para PCM streaming, envie `"stream":true,"response_format":"pcm"`.
