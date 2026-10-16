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


def adapter_state(path, transformer):
    """Normalize this pinned PEFT export and reject missing/mismatched targets."""
    from safetensors.torch import load_file
    raw = load_file(str(path), device="cpu")
    state = {}
    for key, value in raw.items():
        suffix = next((s for s in (".lora_A.default.weight", ".lora_B.default.weight") if key.endswith(s)), None)
        if suffix is None:
            raise ValueError("Unexpected image adapter tensor name")
        target = key[:-len(suffix)]
        module = transformer.get_submodule(target)
        if value.ndim != 2 or len(module.weight.shape) != 2:
            raise ValueError("Image adapter requires linear weight targets")
        if ("lora_A" in suffix and value.shape[1] != module.weight.shape[1]) or (
                "lora_B" in suffix and value.shape[0] != module.weight.shape[0]):
            raise ValueError("Image adapter shape does not match the base model")
        state["transformer." + key.replace(".default.weight", ".weight")] = value
    if not state:
        raise ValueError("Empty image adapter")
    for key, value in state.items():
        if ".lora_A." in key:
            other = state.get(key.replace(".lora_A.", ".lora_B."))
            if other is None or value.shape[0] != other.shape[1]:
                raise ValueError("Image adapter rank/pair mismatch")
        elif key.replace(".lora_B.", ".lora_A.") not in state:
            raise ValueError("Image adapter pair missing")
    return state


def verify_loaded(state, transformer):
    import torch
    for key, expected in state.items():
        target, side, _ = key.removeprefix("transformer.").rsplit(".", 2)
        actual = getattr(transformer.get_submodule(target), side)["visitor"].weight
        if not torch.equal(actual.detach().cpu(), expected.to(dtype=actual.dtype)):
            raise ValueError("Image adapter weights were not loaded exactly")


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
    state = adapter_state(path, pipe.transformer)
    # No permanent fusion: even a failed request must restore the original model.
    try:
        pipe.load_lora_weights(state, adapter_name="visitor")
        verify_loaded(state, pipe.transformer)
        pipe.set_adapters(["visitor"], adapter_weights=[strength])
        yield {"id": adapter_id, "name": spec["name"], "version": spec["version"],
               "sha256": spec["sha256"], "strength": strength}
    finally:
        pipe.unload_lora_weights()
