"""Minimal pinned-checkpoint JumpReLU inference for Gemma Scope 2 residual SAEs.

Formula/normalization verified against the SAELens Gemma 3 checkpoint loader.
No decoder-bias subtraction or activation normalization before encoding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .io import digest, environment, load_dataset, new_directory, read_json, write_json, write_jsonl

REPO = "google/gemma-scope-2-4b-it"
REVISION = "3e94b68be95290aada5b7525cf431d3040f81bb1"


def encode(x, weights):
    pre = x @ weights["w_enc"] + weights["b_enc"]
    return torch.relu(pre) * (pre > weights["threshold"])


def features(run, output, layer=17):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from scipy.sparse import csr_matrix, save_npz
    manifest, rows, arrays = load_dataset(run)
    if (manifest["synthetic"] or manifest["config"]["model_id"] != "google/gemma-3-4b-it" or
        manifest["config"]["revision"] != "093f9f388b31de276ce2de164bdc2081324b9767"):
        raise ValueError("This SAE requires real Gemma 3 4B IT residual activations")
    if manifest["site"] != "resid_post" or manifest["module_path"] != "model.language_model.layers":
        raise ValueError("Checkpoint hook site does not match this extraction")
    if layer not in {9, 17, 22, 29}:
        raise ValueError("This integration supports the verified subset layers: 9,17,22,29")
    folder = f"resid_post/layer_{layer}_width_16k_l0_small"
    cfg_path = hf_hub_download(REPO, f"{folder}/config.json", revision=REVISION)
    cfg = read_json(cfg_path)
    if (cfg["model_name"] != manifest["config"]["model_id"] or cfg["type"] != "sae" or
        cfg["architecture"] != "jump_relu" or cfg["affine_connection"] or
        cfg["hf_hook_point_in"] != f"model.layers.{layer}.output" or
        cfg["hf_hook_point_out"] != cfg["hf_hook_point_in"]):
        raise ValueError("Unsupported or mismatched SAE configuration")
    params_path = hf_hub_download(REPO, f"{folder}/params.safetensors", revision=REVISION)
    weights = {k: v.float() for k, v in load_file(params_path, device="cpu").items()}
    x = arrays[f"layer_{layer}"]
    if weights["w_enc"].shape != (x.shape[1], cfg["width"]):
        raise ValueError("Activation dimension does not match SAE")
    out = new_directory(output)
    sparse_rows, reports = [], []
    with torch.inference_mode():
        for row, vector in zip(rows, x):
            t = torch.from_numpy(vector.copy()).unsqueeze(0)
            z = encode(t, weights)
            reconstructed = z @ weights["w_dec"] + weights["b_dec"]
            values = z[0].numpy()
            if not np.isfinite(values).all():
                raise ValueError("SAE produced non-finite features")
            sparse_rows.append(csr_matrix(values.reshape(1, -1)))
            ids = np.flatnonzero(values)
            top = sorted(ids.tolist(), key=lambda i: -float(values[i]))[:20]
            reports.append({"point_id": row["point_id"], "feature_row": len(reports),
                            "active_features": len(ids),
                            "relative_reconstruction_error": float(torch.linalg.vector_norm(reconstructed - t) /
                                                                      torch.linalg.vector_norm(t).clamp_min(1e-12)),
                            "top_features": [{"feature_id": i, "activation": float(values[i]),
                                              "interpretation": None} for i in top]})
    from scipy.sparse import vstack
    save_npz(out / "features.npz", vstack(sparse_rows))
    write_jsonl(out / "features.jsonl", reports)
    write_json(out / "scope.json", {"repo": REPO, "revision": REVISION, "folder": folder,
               "config": cfg, "weights_sha256": digest(params_path), "config_sha256": digest(cfg_path),
               "source_manifest_sha256": digest(Path(run) / "manifest.json"), "environment": environment(),
               "input_transform": "none", "feature_labels": "uninterpreted; require validation",
               "implementation": "relu(x @ w_enc + b_enc) * (preactivation > threshold)"})
    return out
