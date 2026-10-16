"""Pinned, local-only image adapters. A visitor can select an ID, never a path/URL."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path

CATALOG = json.loads(Path(__file__).with_name("loras.json").read_text(encoding="utf-8"))


def adapter_directory():
    return Path(os.getenv("SOFT_REFUSAL_LORA_DIR", "/workspace/soft-refusal/loras"))


def available_adapters():
    enabled = os.getenv("SOFT_REFUSAL_ENABLE_LORAS", "0") == "1"
    peft = importlib.util.find_spec("peft") is not None
    return [{"id": key, "name": spec["name"], "available": bool(enabled and peft and
             (adapter_directory() / spec["filename"]).is_file())} for key, spec in CATALOG.items()]


@contextmanager
def image_adapter(pipe, adapter_id="none", strength=0.7):
    if adapter_id == "none":
        yield None
        return
    if adapter_id not in CATALOG or os.getenv("SOFT_REFUSAL_ENABLE_LORAS", "0") != "1":
        raise ValueError("Image adapter is not enabled")
    spec = CATALOG[adapter_id]
    path = adapter_directory() / spec["filename"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError("Image adapter checksum mismatch")
    # No permanent fusion: even a failed request must restore the original model.
    try:
        pipe.load_lora_weights(str(path.parent), weight_name=path.name,
                               adapter_name="visitor", local_files_only=True,
                               use_safetensors=True)
        pipe.set_adapters(["visitor"], adapter_weights=[strength])
        yield {"id": adapter_id, "name": spec["name"], "version": spec["version"],
               "sha256": spec["sha256"], "strength": strength}
    finally:
        pipe.unload_lora_weights()
