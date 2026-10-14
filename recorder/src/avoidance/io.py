from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def environment():
    names = ["torch", "transformers", "numpy", "scipy", "scikit-learn", "umap-learn",
             "huggingface-hub", "safetensors", "accelerate", "numba", "pynndescent", "joblib"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {n: importlib.metadata.version(n) for n in names}}


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_dataset(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Run is incomplete; do not analyze partial results.")
    for name, expected in manifest["hashes"].items():
        if digest(path / name) != expected:
            raise ValueError(f"Dataset checksum failed: {name}")
    rows = read_jsonl(path / "steps.jsonl")
    with np.load(path / "activations.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if len({r["point_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate point IDs")
    if [r["activation_row"] for r in rows] != list(range(len(rows))):
        raise ValueError("Activation row order mismatch")
    for value in arrays.values():
        if value.ndim != 2 or len(value) != len(rows) or not np.isfinite(value).all():
            raise ValueError("Invalid activation matrix")
    return manifest, rows, arrays


def measurement_signature(manifest, layer):
    c = manifest["config"]
    return {k: c[k] for k in ["model_id", "revision", "adapter", "dtype", "token_selection"]} | {
        "site": "resid_post", "layer": layer,
        "chat_template_sha256": manifest["chat_template_sha256"],
        # Cloud recordings were computed with the GPU's packages, not this computer's.
        "transformers": (manifest.get("remote_packages") or manifest["environment"]["packages"])["transformers"],
        "synthetic": manifest["synthetic"],
    }
