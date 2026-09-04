#!/usr/bin/env bash
# OX-ALPHA live launcher - one command per venue.
#
#   bash scripts/live.sh live-test      # Dhan connectivity/credential check (safe, exits 0/2)
#   bash scripts/live.sh verify-all     # login-only rehearsal: every venue with keys (safe, no orders)
#   bash scripts/live.sh preflight      # zero-credential readiness check (safe, offline-safe)
#   bash scripts/live.sh dhan           # legacy NSE intraday agent on live Dhan (config.yaml)
#   bash scripts/live.sh choice         # legacy NSE intraday agent on live Choice India (config_choice.yaml)
#   bash scripts/live.sh binance [secs] # promax orchestrator: Dhan equity + Binance live crypto
#   bash scripts/live.sh track          # track record
#   bash scripts/live.sh status         # positions / strategies / recent trades
#   bash scripts/live.sh paper          # paper boot (reverts the config edits)
#   bash scripts/live.sh promax-smoke   # offline promax smoke test
#
# Credentials come from ~/.ox_secrets.env (run bash scripts/setup-live.sh once).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS="${OX_SECRETS_FILE:-$HOME/.ox_secrets.env}"
PYTHON="${PYTHON:-python}"

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
}

require() {
  for var in "$@"; do
    if [ -z "${!var:-}" ]; then
      echo "ERROR: $var is not set. Run 'bash scripts/setup-live.sh' or export it." >&2
      exit 2
    fi
  done
}

load_secrets() {
  if [ ! -f "$SECRETS" ]; then
    echo "ERROR: $SECRETS not found - run 'bash scripts/setup-live.sh' first." >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "$SECRETS"
  set +a
  # The live gates must never depend on a hand-edited file dropping the flag.
  export OX_LIVE_EXECUTION_APPROVED='YES_I_UNDERSTAND_LIVE_TRADING'
}

cmd="${1:-}"
if [ -z "$cmd" ]; then
  usage
  exit 2
fi
shift || true

case "$cmd" in
  dhan)
    load_secrets
    require DHAN_CLIENT_ID DHAN_TOKEN OX_AUDIT_KEY
    sed -i 's/^mode: paper/mode: live/' "$ROOT/config.yaml"
    sed -i 's/^platform: paper/platform: dhan/' "$ROOT/config.yaml"
    echo "config.yaml -> mode: live, platform: dhan (idempotent)"
    cd "$ROOT" && exec "$PYTHON" run.py run
    ;;
  choice)
    load_secrets
    require CHOICE_USER_ID CHOICE_PASSWORD CHOICE_TOTP CHOICE_VENDOR_CODE CHOICE_API_KEY OX_AUDIT_KEY
    sed -i 's/^mode: paper/mode: live/' "$ROOT/config_choice.yaml"
    sed -i 's/^platform: paper/platform: choice/' "$ROOT/config_choice.yaml"
    echo "config_choice.yaml -> mode: live, platform: choice (idempotent)"
    cd "$ROOT" && exec "$PYTHON" run.py run config_choice.yaml
    ;;
  binance)
    load_secrets
    require BINANCE_API_KEY BINANCE_API_SECRET DHAN_CLIENT_ID DHAN_TOKEN OX_AUDIT_KEY
    sed -i 's/^mode: paper/mode: live/' "$ROOT/config_promax.yaml"
    sed -i 's/^platform: paper/platform: dhan/' "$ROOT/config_promax.yaml"
    echo "config_promax.yaml -> mode: live, platform: dhan (idempotent)"
    cd "$ROOT" && exec "$PYTHON" run.py promax "${1:-}"
    ;;
  live-test)
    load_secrets
    require DHAN_CLIENT_ID DHAN_TOKEN
    cd "$ROOT" && exec "$PYTHON" run.py live-test "${1:-0}"
    ;;
  verify-all)
    # Login-only rehearsal against every venue that has keys; never places
    # an order.  Skips venues whose credentials are absent; FAIL exits 2.
    if [ -f "$SECRETS" ]; then
      load_secrets
    fi
    cd "$ROOT" && exec "$PYTHON" run.py venue-check
    ;;
  preflight)
    # Zero-credential readiness: no secrets loaded, no config flips, no
    # network required (probes degrade to SKIP offline).  Run this FIRST,
    # before entering any keys.  Exit 1 when a check FAILs.
    cd "$ROOT" && exec "$PYTHON" run.py preflight
    ;;
  track)
    load_secrets
    cd "$ROOT" && exec "$PYTHON" run.py track-record
    ;;
  status)
    # Booting the agent validates the live config too; load the gate when
    # the secrets file exists, otherwise run paper-mode diagnostics.
    if [ -f "$SECRETS" ]; then
      load_secrets
    fi
    cd "$ROOT" && exec "$PYTHON" run.py status
    ;;
  paper)
    sed -i 's/^mode: live/mode: paper/' "$ROOT/config.yaml"
    sed -i 's/^platform: \(dhan\|choice\)/platform: paper/' "$ROOT/config.yaml"
    sed -i 's/^mode: live/mode: paper/' "$ROOT/config_choice.yaml"
    sed -i 's/^platform: choice/platform: paper/' "$ROOT/config_choice.yaml"
    sed -i 's/^mode: live/mode: paper/' "$ROOT/config_promax.yaml"
    sed -i 's/^platform: dhan/platform: paper/' "$ROOT/config_promax.yaml"
    echo "configs reverted to paper"
    cd "$ROOT" && exec "$PYTHON" run.py run
    ;;
  promax-smoke)
    cd "$ROOT" && OX_PROMAX_AUTO_APPROVE=1 exec "$PYTHON" run.py promax-smoke
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 2
    ;;
esac