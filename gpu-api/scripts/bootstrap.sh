#!/usr/bin/env bash
set -euo pipefail
cd /workspace/soft-refusal
export HF_HOME=/workspace/huggingface
export SOFT_REFUSAL_DATA=/workspace/soft-refusal/runtime
export SOFT_REFUSAL_AUTH_FILE=/workspace/soft-refusal/secrets/auth.json
mkdir -p runtime/images "$HF_HOME"
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
.venv/bin/python -c 'import torch; from diffusers import Flux2KleinPipeline; from transformers import Gemma3ForConditionalGeneration, AutoModelForCausalLM; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; assert torch.cuda.is_bf16_supported(); print("Loader imports and one BF16-capable GPU confirmed:", torch.cuda.get_device_name(0))'
if [[ ! -e secrets/auth.json ]]; then
  .venv/bin/python scripts/init_auth.py
fi
.venv/bin/python -m pip freeze > runtime/rehearsal-environment.txt
printf '%s\n' 'Setup prepared. Authenticate to Hugging Face locally on the Pod if needed, then download_models.py. API submissions remain paused until explicitly enabled.'
