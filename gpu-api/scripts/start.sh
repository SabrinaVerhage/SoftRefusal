#!/usr/bin/env bash
set -euo pipefail
cd /workspace/soft-refusal
export HF_HOME=/workspace/huggingface
export SOFT_REFUSAL_DATA=/workspace/soft-refusal/runtime
export SOFT_REFUSAL_AUTH_FILE=/workspace/soft-refusal/secrets/auth.json
export SOFT_REFUSAL_WEB_ROOT=/workspace/soft-refusal/public
# Set SOFT_REFUSAL_ACCEPT_REQUESTS=1 only for the agreed rehearsal scope.
# Binding externally is a deployment decision; expose only the authenticated
# API through RunPod HTTPS, never the private model process.
export SOFT_REFUSAL_BIND="${SOFT_REFUSAL_BIND:-127.0.0.1}"
exec .venv/bin/python -m soft_refusal.app
