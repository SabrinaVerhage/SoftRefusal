#!/usr/bin/env bash
# Run as root after installing caddy, python3-venv and unattended-upgrades.
# Upload this directory to /root/softrefusal-setup first.
set -euo pipefail
cd -- "$(dirname -- "$0")"
test "$(id -u)" -eq 0
command -v caddy >/dev/null
test -s /root/.ssh/authorized_keys

# The application identity cannot log in and has no administrative privileges.
if ! id softrefusal >/dev/null 2>&1; then
  useradd --system --home-dir /var/lib/softrefusal --shell /usr/sbin/nologin softrefusal
fi
install -d -m 0755 /opt/softrefusal /srv/softrefusal/public
install -d -o softrefusal -g softrefusal -m 0700 /var/lib/softrefusal
install -d -o root -g softrefusal -m 0750 /etc/softrefusal
install -m 0644 index.html /srv/softrefusal/public/index.html
install -m 0644 Caddyfile /etc/caddy/Caddyfile
caddy fmt --overwrite /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile

# Add a small disk-backed cushion for a 1 GiB machine; this is not extra RAM.
if ! test -e /swapfile; then
  fallocate -l 1G /swapfile
  chmod 0600 /swapfile
  mkswap /swapfile
fi
if ! swapon --show=NAME --noheadings | grep -qx /swapfile; then
  swapon /swapfile
fi
if ! grep -q '^/swapfile ' /etc/fstab; then
  printf '/swapfile none swap sw 0 0\n' >> /etc/fstab
fi
printf 'vm.swappiness=10\n' > /etc/sysctl.d/60-softrefusal.conf
sysctl -p /etc/sysctl.d/60-softrefusal.conf

# Retain the verified existing SSH key. Disable password-based network login.
cat > /etc/ssh/sshd_config.d/00-softrefusal.conf <<'SSH'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin prohibit-password
SSH
sshd -t
systemctl reload ssh

cat > /etc/apt/apt.conf.d/20auto-upgrades <<'APT'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT
systemctl enable --now apt-daily.timer apt-daily-upgrade.timer
systemctl enable --now unattended-upgrades

# Bound journal growth on the small disk. No conversation data is in web roots.
install -d /etc/systemd/journald.conf.d
printf '[Journal]\nSystemMaxUse=100M\n' > /etc/systemd/journald.conf.d/softrefusal.conf
systemctl restart systemd-journald
systemctl enable --now caddy
systemctl reload caddy
systemctl is-active caddy
