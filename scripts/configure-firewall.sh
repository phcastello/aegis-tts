#!/usr/bin/env bash
# Show, then explicitly and narrowly apply, the LAN-only UFW rule. This script
# never enables UFW, removes rules, or opens the port to the internet.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

die() {
  printf 'Erro: %s\n' "$*" >&2
  exit 1
}

# Read a simple KEY=value value without executing the .env file as shell code.
dotenv_value() {
  local key="$1"
  local dotenv="$PROJECT_ROOT/.env"
  [[ -f "$dotenv" ]] || return 0
  awk -v wanted="$key" '
    /^[[:space:]]*#/ { next }
    index($0, "=") > 0 {
      candidate = substr($0, 1, index($0, "=") - 1)
      if (candidate == wanted) {
        print substr($0, index($0, "=") + 1)
        exit
      }
    }
  ' "$dotenv" | tr -d '\r'
}

cd "$PROJECT_ROOT"
command -v ufw >/dev/null 2>&1 || die 'ufw não está instalado; nenhuma regra foi aplicada.'

LAN_CIDR="${AEGIS_TTS_LAN_CIDR:-10.1.1.0/24}"
PORT="${AEGIS_TTS_PORT:-}"
[[ -n "$PORT" ]] || PORT="$(dotenv_value AEGIS_TTS_PORT)"
PORT="${PORT:-8001}"

# This service is explicitly scoped to the requested LAN. Refuse a broader or
# substituted network rather than accidentally exposing it elsewhere.
[[ "$LAN_CIDR" == '10.1.1.0/24' ]] || die 'por segurança, este script aceita somente AEGIS_TTS_LAN_CIDR=10.1.1.0/24.'
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1 && PORT <= 65535)) || die 'porta inválida.'

printf 'Regra proposta (não aplicada ainda):\n'
printf '  sudo ufw allow from %s to any port %s proto tcp\n\n' "$LAN_CIDR" "$PORT"
printf 'Estado atual do UFW (somente leitura):\n'
ufw status verbose || true

if [[ ! -t 0 ]]; then
  die 'é necessária confirmação interativa; nenhuma regra foi aplicada.'
fi
read -r -p 'Digite APLICAR para adicionar somente essa regra: ' answer
[[ "$answer" == 'APLICAR' ]] || {
  printf 'Cancelado; nenhuma regra foi aplicada.\n'
  exit 0
}

sudo ufw allow from "$LAN_CIDR" to any port "$PORT" proto tcp
printf '\nEstado do UFW após a adição:\n'
sudo ufw status numbered
