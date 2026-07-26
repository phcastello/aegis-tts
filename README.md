# Aegis TTS

Serviço local de síntese da Aegis, congelado no perfil `AegisVoicev1.0-Q00-W04-P07-M02`.
O único caminho de produção é `docker-compose.yml` → `gateway/Dockerfile` →
`gateway/run-vllm-with-default-reference.sh` → `default_reference_gateway`.

O gateway usa a imagem vLLM-Omni fixada por digest e o modelo
`Qwen3-TTS-12Hz-0.6B-Base`. Ele mantém a referência e a transcrição no mount
local `protected/AegisVoicev1.0`, nunca no request, nos logs ou na imagem.

## Operação

Os diretórios `models/`, `cache/` e a referência privada sob
`protected/AegisVoicev1.0/reference/` são inputs locais, fora do Git. Prepare-os
antes de iniciar e confirme que a transcrição local corresponde ao hash do
manifest protegido.

```bash
docker compose build
docker compose up -d
curl -f http://127.0.0.1:8001/health
curl -f http://127.0.0.1:8001/ready
```

O Compose publica somente `127.0.0.1:8001` por padrão. Consulte
[docs/API.md](docs/API.md) para os contratos HTTP e
[docs/repository-inventory.md](docs/repository-inventory.md) para a decisão de
dependências e retenção.

## Validações locais

```bash
PYTHONPATH=gateway python3 -m aegis_speech validate-profile
PYTHONPATH=gateway python3 -m aegis_speech validate-lexicon
PYTHONPATH=gateway python3 -m aegis_speech run-pronunciation-corpus
PYTHONPATH=gateway python3 -m aegis_speech run-golden-regression
pytest -q
```

`run-golden-regression` é o comando de integridade já existente: ele revalida
o perfil protegido. A comparação acústica/auditiva requer o corpus e os assets
privados locais; não há WAV privado no repositório.

## Smoke operacional

Com o serviço pronto, execute um smoke sem gravar áudio no repositório:

```bash
python3 scripts/smoke-test.py
```

Ele verifica `/health`, `/ready`, métricas e fila, OpenAI WAV, Aegis WAV,
PCM streaming, SSE e cancelamento explícito. Um WAV de 44 bytes é tratado
como falha: ele contém apenas o cabeçalho e deve ser investigado antes da
integração da Aegis. Para outra porta, use
`python3 scripts/smoke-test.py --base-url http://127.0.0.1:PORTA`.

## Estrutura

```text
gateway/                         runtime, API, fila, normalização e DSP
config/pronunciation/            corpus de normalização revisável
config/speech/                   configuração de serviço documentada
protected/AegisVoicev1.0/        receita, proveniência, hashes e código congelado
tests/                           unitários, contrato e opt-in GPU
docs/                            API, arquitetura e inventário
scripts/audio-info.py            inspeção operacional somente-leitura
```

## Rollback e histórico experimental

O rollback suportado é a tag `aegis-tts-production-baseline-v1`, no commit
`fc1b235eadd5ccededbc8d53a535ac3d737402a1`. Ela preserva a árvore anterior;
não há Compose eager alternativo mantido sem uma validação reproduzível.

```bash
git switch --detach aegis-tts-production-baseline-v1
git worktree add ../aegis-tts-history aegis-tts-production-baseline-v1
git show aegis-tts-production-baseline-v1:diagnostics/q01/README.md
```

Os experimentos concluídos foram removidos da branch de produção. Nenhuma
funcionalidade nova, dependência nova ou alteração acústica foi introduzida.
