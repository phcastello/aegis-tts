# Inventário de produção

Baseline: `aegis-tts-production-baseline-v1` (`fc1b235eadd5ccededbc8d53a535ac3d737402a1`).

| Caminho | Classificação | Referências / uso | Runtime/build/test/rollback | Ação | Risco / validação |
| --- | --- | --- | --- | --- | --- |
| `gateway/` | production-runtime | Compose, Dockerfile e imports Python | todos / não rollback | preservado | testes + Compose + smoke |
| `gateway/voxcpm_*`, `gateway/aegis_speech/experimental_text.py` | experimental-runtime | somente feature flag `AEGIS_TTS_BACKEND=voxcpm2` e JSON do operador | CPU/fakes; GPU/HTTP real pendentes | opcional, sem instalar dependências na imagem original | perfil v2 distinto; Qwen padrão; [receita](v2.0/voxcpm-gateway-experimental-20261103.md) |
| `config/pronunciation/`, `config/speech/` | production-config | CLI e documentação | runtime/teste | preservado | corpus e léxico |
| `protected/AegisVoicev1.0/` | protected-profile | mount `/protected`, hashes, gateway | runtime/regressão | preservado | `validate-profile`; referência local fora do índice |
| `tests/test_aegis_*`, `tests/test_vllm_*`, `tests/test_gpu_real.py` | production-test | pytest | teste | preservados/atualizados | unitário e opt-in GPU |
| `scripts/audio-info.py` | operational-tool | ferramenta read-only independente | operação | preservado | execução local |
| `app/`, Dockerfile raiz e requirements raiz | dead-code | somente Compose/testes antigos | nenhum | removidos | Compose e pytest |
| `diagnostics/` e Dockerfile de patch | historical-experiment / generated-output | apenas mount amplo e diagnóstico Q01 | nenhum | removidos | mount removido; build limpo |
| `modules/aegis-robotic-voice-lab/` | historical-experiment | nenhuma importação do gateway; cópia hashada existe no perfil | nenhum | removido | hashes v1.0 e testes DSP |
| `tools/`, configs GLaDOS/Graillon | historical-experiment | somente testes e scripts laboratoriais | nenhum | removidos | pytest sem testes laboratório |
| `docker-compose.*` alternativos | dead-code | runtime eager/app ou diagnóstico | rollback não validado | removidos | Compose único |
| `protected/AegisVoicev0.1/` | historical-experiment | sem referência da produção | nenhum | removido | tag baseline |
| modelos, cache, WAVs, transcript e outputs | private-artifact / cache | mounts locais | runtime local | fora do índice | `.gitignore`, clone limpo |

## Mapa de dependências

`docker-compose.yml` constrói `gateway/Dockerfile`, que copia somente
`gateway/` e instala `gateway/requirements-runtime.txt`. O entrypoint inicia
vLLM com `protected/AegisVoicev1.0/config/q00-vllm-deploy-exact.yaml` e o
gateway Python. Este importa `aegis_speech` (normalização, segmentação e fila)
e `voice_pipeline` (DSP). O gateway carrega apenas os assets verificados pelo
manifest dentro do mount `/protected`. Testes importam os módulos de gateway
ou verificam contratos estáticos; `test_gpu_real.py` é opt-in.

Não há import dinâmico, subprocesso, `PYTHONPATH`, mount, configuração,
manifest ou script de produção que alcance `app/`, `diagnostics/`, `tools/`
ou o laboratório removido. A única referência histórica necessária à seleção
W04/P07/M02 é a cópia congelada e hashada em `protected/AegisVoicev1.0/code/`.
Os manifests e a proveniência imutáveis desse perfil também registram paths
absolutos do laboratório e do perfil v0.1 como evidência da seleção histórica;
eles não são carregados pelo runtime e não podem ser editados sem invalidar os
hashes aprovados.

## Histórico experimental

Os experimentos retirados da branch continuam recuperáveis pela tag baseline:

```bash
git show aegis-tts-production-baseline-v1:modules/aegis-robotic-voice-lab/README.md
git worktree add ../aegis-tts-history aegis-tts-production-baseline-v1
```
