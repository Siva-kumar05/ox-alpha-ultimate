#!/usr/bin/env bash
# EC2 rollback + clean bootstrap for OX-ALPHA PRIME.
#
#   ./scripts/ec2_rollback_bootstrap.sh <PATH_TO_NEW_CODE_TARBALL_OR_DIR>
#
# What it does:
#   1. STOPS any running agent processes (systemd units, pm2, tmux sessions,
#      cron entries) that belong to previous uploaded versions.
#   2. WIPES every previous copy of the agent it can find
#      (~/*ox*, ~/ox*, /opt/ox*, /srv/ox*, old tarballs, nano/.swp backups).
#   3. SCRUBS credentials you previously pasted into the console:
#      bash_history lines containing tokens (shredded), ~/.env-style files,
#      nano swap files.  Tokens live ONLY in ~/.ox_secrets.env going forward.
#   4. Extracts the fresh code into ~/ox-alpha (or copies the dir), creates
#      the venv, installs requirements, writes an empty ~/.ox_secrets.env
#      template with 600 perms, and runs the offline test suite.
#
# Run from the EC2 shell as the ubuntu/ec2-user login.  Review the rm list
# before running — it is intentionally aggressive.
set -euo pipefail

NEW_CODE="${1:?usage: $0 <tarball.gz | source directory>}"
HOME_DIR="$HOME"
AGENT_HOME="$HOME_DIR/ox-alpha"
SECRETS="$HOME_DIR/.ox_secrets.env"

# Input guard: NEW_CODE is operator-supplied and reaches tar/cp.  Restrict it
# to a single path token that exists under /home, /tmp or /opt — no globs,
# spaces, semicolons or command metacharacters.
case "$NEW_CODE" in
  *[';&|`$()<>"'\''\ ']*|*..*|"")
    echo "rejected suspicious path: $NEW_CODE" >&2; exit 2 ;;
esac
case "$(readlink -f "$NEW_CODE" 2>/dev/null || echo missing)" in
  "$HOME_DIR"/*|/tmp/*|/opt/*) : ;;
  *) echo "path must live under /home, /tmp or /opt: $NEW_CODE" >&2; exit 2 ;;
esac

echo "==> [1/6] stopping previous versions"
sudo systemctl stop oxalpha promax 2>/dev/null || true
sudo systemctl disable oxalpha promax 2>/dev/null || true
sudo rm -f /etc/systemd/system/oxalpha.service /etc/systemd/system/promax.service
sudo systemctl daemon-reload 2>/dev/null || true
pm2 delete oxalpha promax 2>/dev/null || true
tmux kill-session -t oxalpha 2>/dev/null || true
tmux kill-session -t promax 2>/dev/null || true
crontab -l 2>/dev/null | grep -viE 'oxalpha|promax|ox-alpha' | crontab - || true
pkill -f 'run\.py (run|promax|train)' 2>/dev/null || true

echo "==> [2/6] wiping previous uploaded copies"
rm -rf "$HOME_DIR"/oxalpha* "$HOME_DIR"/ox-alpha* "$HOME_DIR"/promax* \
       /opt/ox* /srv/ox* \
       "$HOME_DIR"/.ox-alpha 2>/dev/null || true

echo "==> [3/6] scrubbing pasted credentials from shell history & backups"
if [ -f "$HOME_DIR/.bash_history" ]; then
  grep -viE 'DHAN_TOKEN|DHAN_ACCESS_TOKEN|DHAN_CLIENT_ID|DHAN_PIN|DHAN_TOTP|FINX|eyJ|access-token' \
    "$HOME_DIR/.bash_history" > "$HOME_DIR/.bash_history.clean" || true
  shred -u "$HOME_DIR/.bash_history" 2>/dev/null || rm -f "$HOME_DIR/.bash_history"
  mv "$HOME_DIR/.bash_history.clean" "$HOME_DIR/.bash_history"
fi
history -c 2>/dev/null || true
rm -f "$HOME_DIR"/.*.swp "$HOME_DIR"/*.*.swp 2>/dev/null || true   # nano/vi secrets
# legacy env files that may hold old keys
for f in "$HOME_DIR"/.env "$AGENT_HOME"/.env "$AGENT_HOME"/.secrets/*; do
  [ -f "$f" ] && shred -u "$f" 2>/dev/null || true
done

echo "==> [4/6] installing fresh code -> $AGENT_HOME"
mkdir -p "$AGENT_HOME"
case "$NEW_CODE" in
  *.tar.gz|*.tgz) tar -xzf "$NEW_CODE" -C "$AGENT_HOME" --strip-components=1 ;;
  *) cp -a "$NEW_CODE"/. "$AGENT_HOME"/ ;;
esac

echo "==> [5/6] venv + secrets template"
cd "$AGENT_HOME"
python3 -m venv .venv
./.venv/bin/pip install -q -r requirements.txt
if [ ! -f "$SECRETS" ]; then
  umask 077
  cat > "$SECRETS" <<'EOF'
# chmod 600. Fill per session; rotate the 24h Dhan key daily.
export DHAN_TOKEN=""
export DHAN_CLIENT_ID=""
# Optional integrations (leave blank if unused)
export OX_TG_BOT_TOKEN=""
export OX_TG_CHAT_ID=""
export OX_X_BEARER=""
EOF
  chmod 600 "$SECRETS"
  echo "    wrote $SECRETS (fill it: nano $SECRETS)"
fi
chmod 600 "$SECRETS"

echo "==> [6/6] offline test suite"
./.venv/bin/python -m pytest tests/ -q

cat <<EOF

NEXT (from $AGENT_HOME):
  source ~/.ox_secrets.env                      # today's Dhan key
  ./.venv/bin/python run.py live-test 60        # read-only live checks + PRIME session
  ./.venv/bin/python run.py promax              # full multi-agent paper run on live quotes
EOF
