"""Download exact, safetensors-only snapshots before opening the GPU API."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soft_refusal.config import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-hermes", action="store_true")
    parser.add_argument("--temporary-token-file", type=Path,
                        help="Read an authorized temporary token into memory and remove its file immediately.")
    args = parser.parse_args()
    # Some RunPod images enable the optional legacy hf-transfer client without
    # installing it. Use the official Hub client's normal transfer path instead.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from huggingface_hub import HfApi, snapshot_download
    token = None
    if args.temporary_token_file:
        token = args.temporary_token_file.read_text(encoding="utf-8").strip()
        args.temporary_token_file.unlink()
        if not token:
            raise SystemExit("Temporary authentication file was empty.")
    root = Path(os.environ.get("HF_HOME", "/workspace/huggingface"))
    root.mkdir(parents=True, exist_ok=True)
    selected = ["gemma", "flux"] + (["hermes"] if args.include_hermes else [])
    if shutil.disk_usage(root).free < (90 if args.include_hermes else 55) * 1024**3:
        raise SystemExit("Insufficient free persistent storage for model downloads and working headroom.")
    manifest = {}
    for key in selected:
        spec = MODELS[key]
        info = HfApi(token=token).model_info(spec["repo"], revision=spec["revision"])
        if info.sha != spec["revision"]:
            raise SystemExit("Model revision mismatch")
        print(f"Downloading pinned {spec['repo']} (no quantization).")
        patterns = (["model_index.json", "LICENSE.md", "README.md", "scheduler/*", "tokenizer/*",
                     "text_encoder/*.json", "text_encoder/*.safetensors", "transformer/*.json",
                     "transformer/*.safetensors", "vae/*.json", "vae/*.safetensors"] if key == "flux" else
                    ["*.json", "*.safetensors", "*.jinja", "*.txt", "tokenizer.model", "README.md", "LICENSE*"])
        location = snapshot_download(spec["repo"], revision=spec["revision"], allow_patterns=patterns,
                                     ignore_patterns=["*.bin", "*.pt", "*.pth", "*.py"], max_workers=4, token=token)
        manifest[key] = {**spec, "cache_path": location}
    (root / "soft-refusal-downloads.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Snapshots downloaded. GPU fit and speed are still unverified.")


if __name__ == "__main__":
    main()
