from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .io import digest, environment, new_directory, read_json, read_jsonl, write_json, write_jsonl


def validate_conversations(conversations):
    seen = set()
    for c in conversations:
        cid = c["conversation_id"]
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("Conversation IDs must be unique nonempty strings")
        seen.add(cid)
        if not isinstance(c.get("group_id"), str) or not c["group_id"]:
            raise ValueError("Provide a group_id linking related/paraphrased conversations")
        if not c["turns"] or any(not isinstance(t, str) or not t.strip() for t in c["turns"]):
            raise ValueError("turns must be nonempty user strings")
    if not conversations:
        raise ValueError("Empty dataset")


def collect(config_path, input_path, output, synthetic=False):
    c = read_json(config_path)
    if not re.fullmatch(r"[0-9a-f]{40}", c["revision"]):
        raise ValueError("Pin the model to a full 40-character commit SHA")
    if c["max_new_tokens"] < 1 or c["max_input_tokens"] < 1:
        raise ValueError("Token limits must be positive")
    conversations = read_jsonl(input_path)
    validate_conversations(conversations)
    out = new_directory(output)
    manifest = {"schema_version": 1, "status": "incomplete", "synthetic": synthetic,
                "created_utc": datetime.now(timezone.utc).isoformat(), "config": c,
                "input_sha256": digest(input_path), "environment": environment(),
                "site": "resid_post", "layer_numbering": "zero_based", "storage_dtype": "float32"}
    manifest["source_sha256"] = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    manifest["generation_overrides"] = {"do_sample": False, "num_beams": 1,
                                        "max_new_tokens": c["max_new_tokens"], "use_cache": True}
    manifest["capture_execution"] = "synthetic" if synthetic else ("generation_prefill" if c["token_selection"] == "last_prompt" else "after_generation")
    write_json(out / "manifest.json", manifest)
    write_jsonl(out / "input.jsonl", conversations)
    if synthetic:
        adapter = None
        template = "SYNTHETIC FIXTURE: no tokenizer or language model"
        manifest["module_path"] = "synthetic"
    else:
        from .model import HFAdapter
        adapter = HFAdapter(c)
        template = adapter.tokenizer.get_chat_template()
        manifest["module_path"] = adapter.module_path
        write_json(out / "model_config.json", adapter.model.config.to_dict())
        write_json(out / "generation_config.json", adapter.model.generation_config.to_dict())
        import torch
        manifest["hardware"] = {"cuda": torch.version.cuda,
                                "device": torch.cuda.get_device_name() if c["device"] == "cuda" else "cpu"}
    (out / "chat_template.txt").write_text(template, encoding="utf-8")
    manifest["chat_template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
    rows, collected = [], {f"layer_{i}": [] for i in c["layers"]}
    rng = np.random.default_rng(c["seed"])
    for conversation in conversations:
        messages = []
        for turn, prompt in enumerate(conversation["turns"]):
            messages.append({"role": "user", "content": prompt})
            if synthetic:
                result = {"response": "SYNTHETIC plumbing fixture; no model response.",
                          "rendered_prompt": template, "prompt_token_ids": [], "response_token_ids": [],
                          "phase": "pre_response" if c["token_selection"] == "last_prompt" else "post_response",
                          "token_index": None, "token_id": None, "token_text": None,
                          "input_token_count": None, "generated_token_count": None,
                          "ended_with_eos": None, "hit_token_limit": None}
                vectors = {i: rng.standard_normal(32).astype("float32") for i in c["layers"]}
            else:
                result, vectors = adapter.step(messages)
            point_id = hashlib.sha256(f"{conversation['conversation_id']}\0{turn}".encode()).hexdigest()[:24]
            row = {"point_id": point_id, "activation_row": len(rows),
                   "conversation_id": conversation["conversation_id"], "group_id": conversation["group_id"],
                   "turn_id": turn, "prompt": prompt, "messages": list(messages),
                   "outcome": "unlabeled", "label_source": None,
                   "model_id": c["model_id"], "model_revision": c["revision"],
                   "layers": c["layers"], "site": "resid_post", "token_selection": c["token_selection"],
                   "synthetic": synthetic, **result}
            for layer, vector in vectors.items():
                if vector.ndim != 1 or not np.isfinite(vector).all():
                    raise ValueError("Non-finite or non-vector activation")
                collected[f"layer_{layer}"].append(vector)
            rows.append(row)
            messages.append({"role": "assistant", "content": result["response"]})
            # Save incremental textual evidence, but never mark a partial run complete.
            write_jsonl(out / "steps.jsonl", rows)
            print(f"Captured {conversation['conversation_id']} turn {turn}", flush=True)
    np.savez_compressed(out / "activations.npz", **{k: np.stack(v) for k, v in collected.items()})
    with (out / "annotations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["point_id", "outcome", "annotator", "split", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"point_id": row["point_id"], "outcome": "unlabeled"})
    files = ["input.jsonl", "steps.jsonl", "activations.npz", "chat_template.txt"]
    if not synthetic:
        files += ["model_config.json", "generation_config.json"]
    manifest.update(status="complete", point_count=len(rows),
                    hashes={name: digest(out / name) for name in files})
    write_json(out / "manifest.json", manifest)
    return out
