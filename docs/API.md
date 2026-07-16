# API de produção — Aegis TTS / vLLM-Omni

Base URL: `http://10.1.1.47:8001`

O serviço usa vLLM-Omni 0.22.0 e `Qwen/Qwen3-TTS-12Hz-0.6B-Base`. O endpoint
de síntese é compatível com OpenAI: `POST /v1/audio/speech`.

## Referência padrão

Se `ref_audio` não for enviado, o gateway usa automaticamente a referência
aprovada `giselle_AegisVoice3.wav` (aproximadamente 28 segundos) e o respectivo
arquivo `giselle_AegisVoice3.txt`. Ambos ficam no host, montados somente para
leitura, e nunca são devolvidos pela API ou registrados nos logs.

Se enviar `ref_audio`, envie também o `ref_text` correspondente. O gateway não
mistura um áudio fornecido pelo cliente com a transcrição padrão. Não envie o
campo `voice`: no modo Base a referência ICL é a seleção de voz.

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

Os valores omitidos passam a ser `task_type: "Base"`, `language: "Portuguese"`,
`x_vector_only_mode: false`, `response_format: "wav"` e o modelo 0.6B local.
O servidor usa o preset validado: temperatura 0,8, top-p 0,9, top-k 50,
máximo de 1024 tokens, seed 41 e `min_tokens: 0`.

## Síntese com referência própria

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

`ref_audio` pode ser uma data URL/base64 compatível com o runtime. A
transcrição deve corresponder ao áudio; não envie somente um speaker embedding.

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

O PCM retornado é mono, 16-bit, 24 kHz. Para analisá-lo como WAV:

```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i resposta.pcm resposta.wav
```

## Health e modelo carregado

```bash
curl --fail http://10.1.1.47:8001/health
curl --fail http://10.1.1.47:8001/v1/models
```

`/health` responde somente quando o backend vLLM-Omni estiver disponível. O
modelo retornado em `/v1/models` é `/models/Qwen3-TTS-12Hz-0.6B-Base`.

## Limites e compatibilidade

- A rota antiga `/v1/synthesize` não faz parte do runtime padrão atual.
- O benchmark aprovado mediu RTF mediano em torno de 0,52; o desempenho de uma
  chamada individual varia com o texto e a carga da GPU.
- Não defina `min_tokens` positivo: ele atinge um bug conhecido na combinação
  Qwen3-TTS Base/vLLM-Omni 0.22.0 e pode causar erro CUDA.
