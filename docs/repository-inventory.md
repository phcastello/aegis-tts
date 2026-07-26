# Inventário do repositório — 2026-07-26

| Caminho | Propósito | Produção | Arquivável | Privado | Decisão/responsável |
|---|---|---:|---:|---:|---|
| gateway/ | Gateway HTTP, scheduler e DSP de produção | sim | não | apenas por mount | equipe Aegis, 2026-07-26 |
| protected/AegisVoicev1.0/ | Evidência, referência e configuração congelada | sim | não | sim | somente leitura, equipe de voz |
| config/pronunciation/ | corpus e fonte revisável do léxico | sim | não | não | equipe Aegis |
| config/speech/ | limites operacionais documentados | sim | não | não | equipe Aegis |
| modules/aegis-robotic-voice-lab/ | laboratório que selecionou W04/P07/M02 | não | sim, preservar | pode conter resultados locais | histórico de voz |
| tools/, diagnostics/ | diagnósticos e experimentos Q00/GLaDOS | não | sim, preservar | outputs locais | laboratório |
| experiments/ | experiências não montadas no runtime | não | sim, preservar | alguns inputs | laboratório |
| app/ | serviço anterior/alternativo, não invocado pelo Compose atual | não | sim após migração explícita | não conhecido | requer decisão futura |
| tests/ | testes do gateway e laboratórios | sim | não | não | equipe Aegis |
| docs/ | operação e contratos | sim | não | não | equipe Aegis |

Nenhum diretório foi movido nesta alteração: há imports, composições e evidências
locais pendentes de uma migração explícita. Os WAVs privados permanecem fora do
Git pelo arquivo de ignorados; o Compose monta o perfil protegido somente como
leitura.
