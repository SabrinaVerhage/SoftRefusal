#!/usr/bin/env bash
# Usage: softrefusal-deploy [origin/main|previous-commit]
# Run as root. GitHub is read-only on the server; no GitHub credential is needed.
set -euo pipefail
test "$(id -u)" -eq 0
exec 9>/run/lock/softrefusal-deploy.lock
flock -n 9 || { echo 'Another deployment is running.'; exit 1; }
base=/opt/softrefusal
repo="$base/repository"
git -C "$repo" fetch origin
commit=$(git -C "$repo" rev-parse --verify --end-of-options "${1:-origin/main}^{commit}")
release="$base/releases/$commit"
mkdir -p "$base/releases" "$base/venvs"
if ! test -d "$release"; then
  mkdir "$release"
  git -C "$repo" archive "$commit" | tar -x -C "$release"
fi
test -f "$release/hosted/requirements.txt"
envkey=$(sha256sum "$release/hosted/requirements.txt" | cut -d' ' -f1)
venv="$base/venvs/$envkey"
if ! test -x "$venv/bin/python"; then python3 -m venv "$venv"; fi
"$venv/bin/pip" install -r "$release/hosted/requirements.txt"
ln -sfn "$venv" "$release/.venv"
printf '%s\n' "$commit" > "$release/REVISION"
PYTHONPATH="$release/recorder/src:$release" "$venv/bin/python" -c 'from hosted.app import app; from avoidance.remote import RemoteAdapter'
caddy validate --config "$release/deploy/greenhost/Caddyfile.app" --adapter caddyfile

# Refuse to interrupt a live recording during routine updates.
if systemctl is-active --quiet softrefusal; then
  active=$(curl -fsS -H 'Origin: https://softrefusal.sabrinaverhage.com' -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8780/internal/drain | python3 -c 'import json,sys; print(json.load(sys.stdin)["active"])')
  if test "$active" -gt 0; then
    curl -fsS -H 'Origin: https://softrefusal.sabrinaverhage.com' -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8780/internal/resume >/dev/null
    echo 'Visitors are recording. Retry once the queue is idle.'; exit 1
  fi
fi
previous=$(readlink -f "$base/current" || true)
cp /etc/caddy/Caddyfile "$base/Caddyfile.previous"
rollback() {
  echo 'Deployment failed; restoring previous release.'
  if test -n "$previous" && test -d "$previous"; then
    ln -sfn "$previous" "$base/current.next"
    mv -Tf "$base/current.next" "$base/current"
    install -m 0644 "$previous/deploy/greenhost/softrefusal.service" /etc/systemd/system/softrefusal.service
    systemctl daemon-reload
    systemctl restart softrefusal || true
  else
    systemctl stop softrefusal || true
  fi
  cp "$base/Caddyfile.previous" /etc/caddy/Caddyfile
  systemctl reload caddy || true
}
trap rollback ERR
ln -sfn "$release" "$base/current.next"
mv -Tf "$base/current.next" "$base/current"
install -m 0644 "$release/deploy/greenhost/softrefusal.service" /etc/systemd/system/softrefusal.service
systemctl daemon-reload
systemctl enable softrefusal
systemctl restart softrefusal
healthy=0
for attempt in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8780/healthz | python3 -c 'import json,sys; assert json.load(sys.stdin)["version"] == sys.argv[1]' "$commit" 2>/dev/null; then healthy=1; break; fi
  sleep 1
done
test "$healthy" -eq 1
install -m 0644 "$release/deploy/greenhost/Caddyfile.app" /etc/caddy/Caddyfile
caddy fmt --overwrite /etc/caddy/Caddyfile
systemctl reload caddy
curl -fsS --resolve softrefusal.sabrinaverhage.com:443:127.0.0.1 https://softrefusal.sabrinaverhage.com/healthz
install -m 0755 "$release/deploy/greenhost/deploy.sh" /usr/local/sbin/softrefusal-deploy
trap - ERR
printf '\nDeployed %s\nPrevious release: %s\n' "$commit" "$previous"
