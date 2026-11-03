# Preparação da entrega VoxCPM2

Há um caminho direto de uso com uvicorn, sem reconstruir o laboratório ou
instalar outro ambiente. A sequência concreta para este host está no runbook
privado `/var/tmp/aegis-v2-work/voxcpm-gateway/handoff/RUNBOOK.md`, acompanhada
de `preparation-receipt.json` e da configuração privada de engenharia.

**Os comandos de uso direto/rollback não foram executados nesta preparação.**
O binário/argv/env/config vêm da prova HTTP já concluída; a sequência manual
ainda precisa de janela exclusiva. O treino segue ativo. CP400 é somente um
controle; pesos finais não foram selecionados e frontend numérico/reserva320ms
continuam provisórios para a entrega.

| Uso | Ferramenta existente | Condição |
| --- | --- | --- |
| Servir a voz experimental | uvicorn no Python F5 com overlay API estreito | Janela GPU exclusiva; processo foreground |
| WAV completo e PCM progressivo | rotas OpenAI-compatible e Aegis existentes | Ready concluído; PCM16LE mono24k |
| Player | página do próprio gateway | Loopback8002 ou túnel do operador |
| Fila/SSE/cancelamento | `scripts/smoke-test.py` | Smoke de contrato, não benchmark |
| Benchmark isolado | `training/v2_voxcpm_gateway_probe.py` | Exclusivamente GPU worker; encerra o servidor |
| Restaurar Qwen | container original preservado; snapshots Compose/env | GPU liberada; imagem/flags originais conferidos |

O runner restrito ao GPU worker não é um launcher permanente: ele mede e mata
apenas seu filho. O comando direto usa o runtime existente, com checks de GPU
livre/RAM/cache, mas não coordena a fila de treinos nem faz o inventário global
de40GB do runner. O runbook inclui preflight e exige exclusividade operacional.
Foreground evita instalar um gerenciador ou criar logs locais sem limite.

Devem ser mantidos o Python F5, overlay verificado, fonte upstream pinada,
base/AudioVAE, headers de compilação, cache, adapter+sidecar estáveis e config.
Os arquivos grandes foram verificados por tamanho/header nesta preparação;
seus SHA integrais já têm recibos e são novamente exigidos pelo loader. Nenhum
Torch/modelo foi importado ou carregado nesta tarefa.

O perfil nativo v2 é distinto do protegido v1. Um cliente que fixe o nome v1
deve continuar no Qwen ou usar explicitamente a instância experimental; não se
rebatiza Vox como v1. A volta ao Qwen prefere iniciar o container exato já
preservado, cuja imagem e parâmetros coincidem com o baseline medido. Usar
Compose sem o env salvo poderia alterar22×22/corte/reserva; não há build novo
na receita de rollback.

Pendências reais: checkpoint final estável e seus hashes, processo novo com
esses pesos, smoke direto, benchmark/ASR/identidade e decisão reservada. A
instância experimental só poderá ficar como serviço final após esses gates.
Este documento não publica config privada, áudio ou transcrições, nem abre o
teste final de seis frases ou os50 trechos reservados.
