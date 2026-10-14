"""Recorded conversations → one map file for the viewer in ../map.

Combines the latest checkpoint of every conversation in a recordings folder and
fits a separate 3D UMAP per recorded layer (cosine on raw residual vectors, the
same settings as `reduce`). Exploratory and in-sample: axes, distances and paths
are not refusal measurements.

With human labels it also places every turn on the "semantic donut": distance from
the centre follows a refusal axis (mean refused minus mean answered direction, as in
difference-of-means refusal-direction work), the angle follows what remains of the
state once that axis is removed. The axis is defined by the same labelled turns it
displays, so labelled turns land where their label says by construction."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .io import load_dataset, read_json, read_jsonl

OUTCOMES = ("answered", "redirected", "refused", "unclear", "unlabeled")


def read_labels(directory):
    """Latest human label per turn, from the recording room's append-only labels.jsonl."""
    path = Path(directory) / "labels.jsonl"
    return {row["point_id"]: row["outcome"] for row in read_jsonl(path)} if path.exists() else {}


def collect_recordings(runs):
    """Rows and per-layer matrices from every completed conversation, oldest first."""
    sessions = []
    for path in Path(runs).glob("*/session.json"):
        meta = read_json(path)
        if meta.get("latest"):
            sessions.append((meta["created_utc"], path.parent, meta))
    if not sessions:
        raise ValueError(f"No completed conversations in {runs}")
    rows, matrices, signature = [], {}, None
    for number, (_, directory, meta) in enumerate(sorted(sessions, key=lambda s: s[0]), start=1):
        manifest, session_rows, arrays = load_dataset(directory / meta["latest"])
        config = manifest["config"]
        current = (config["model_id"], config["revision"], tuple(config["layers"]), config["token_selection"])
        if signature is None:
            signature = current
        elif current != signature:
            # 4B and 12B snapshots (or different layers) cannot share one map.
            raise ValueError(f"{directory.name} was recorded with different settings; map one model and layer set at a time")
        labels = read_labels(directory)
        for row in session_rows:
            rows.append({**row, "conversation_number": number, "conversation_title": meta.get("title", ""),
                         "outcome": labels.get(row["point_id"], "unlabeled")})
        for key, matrix in arrays.items():
            matrices.setdefault(key, []).append(matrix)
    return rows, {key: np.vstack(parts) for key, parts in matrices.items()}, signature


def fit_layer(x, seed, neighbors, min_dist):
    from umap import UMAP
    if len(x) < 6 or len(np.unique(x, axis=0)) < 6:
        raise ValueError("Need at least six distinct recorded turns for a 3D UMAP")
    if (np.linalg.norm(x, axis=1) == 0).any():
        raise ValueError("Cosine distance is undefined for zero vectors")
    settings = dict(n_components=3, n_neighbors=min(neighbors, len(x) - 1), min_dist=min_dist,
                    metric="cosine", init="random", random_state=seed, transform_seed=seed, n_jobs=1)
    coords = UMAP(**settings).fit_transform(x)
    if not np.isfinite(coords).all():
        raise ValueError("UMAP produced non-finite coordinates")
    return coords, settings


def semantic_layout(x, outcomes):
    """Donut coordinates for each turn: refusal score (radius), angle and lift. None + reason without labels."""
    outcomes = np.asarray(outcomes)
    core_label = "refused" if (outcomes == "refused").any() else "redirected"
    core, outer = outcomes == core_label, outcomes == "answered"
    if not core.any() or not outer.any():
        return None, "Label at least one reply as refused (or redirected) and one as answered in the recording room."
    # Directions only: residual norms vary with token position and conversation length.
    u = x / np.linalg.norm(x, axis=1, keepdims=True)
    axis = u[core].mean(0) - u[outer].mean(0)
    axis /= np.linalg.norm(axis)
    raw = u @ axis
    low, high = raw[outer].mean(), raw[core].mean()
    score = (raw - low) / (high - low)  # 0 = like the answered turns, 1 = like the core turns.
    rest = u - np.outer(raw, axis)
    rest -= rest.mean(0)
    _, _, vt = np.linalg.svd(rest, full_matrices=False)
    pcs = np.zeros((len(x), 3))
    pcs[:, :min(3, len(vt))] = rest @ vt[:3].T
    lift = pcs[:, 2] / (np.abs(pcs[:, 2]).max() or 1)
    return {"score": score, "angle": np.arctan2(pcs[:, 1], pcs[:, 0]), "lift": lift, "core_label": core_label,
            "counts": {k: int((outcomes == k).sum()) for k in OUTCOMES}}, None


def export_map(runs, out, layer=None, seed=42, neighbors=10, min_dist=0.1):
    rows, matrices, (model_id, revision, layers, _) = collect_recordings(runs)
    layers = list(layers)
    default = layer if layer is not None else layers[len(layers) // 2 - 1 if len(layers) > 1 else 0]
    if default not in layers:
        raise ValueError(f"Layer {default} was not recorded; choose one of {layers}")
    coordinates, settings = {}, None
    for n in layers:
        coordinates[n], settings = fit_layer(matrices[f"layer_{n}"], seed, neighbors, min_dist)
    semantic, reason = semantic_layout(matrices[f"layer_{default}"], [r["outcome"] for r in rows])
    previous, points = {}, []
    for i, row in enumerate(rows):
        cid = row["conversation_id"]
        points.append({
            "point_id": row["point_id"], "conversation_id": cid, "turn_id": row["turn_id"],
            "previous_point_id": previous.get(cid),
            "conversation_number": row["conversation_number"], "conversation_title": row["conversation_title"],
            "prompt": row["prompt"], "response": row["response"], "outcome": row.get("outcome", "unlabeled"),
            "hit_token_limit": row.get("hit_token_limit", False),
            "topic": f"conversation {row['conversation_number']}",
            "layers": {str(n): [round(float(v), 6) for v in coordinates[n][i]] for n in layers},
        })
        previous[cid] = row["point_id"]
        x, y, z = points[-1]["layers"][str(default)]
        points[-1].update(x=x, y=y, z=z)
        if semantic:
            points[-1].update(core_score=round(float(semantic["score"][i]), 6),
                              angle=round(float(semantic["angle"][i]), 6), lift=round(float(semantic["lift"][i]), 6))
    payload = {
        "name": "Your recordings", "kind": "recordings", "synthetic": False,
        "model_id": model_id, "revision": revision, "layers": layers, "default_layer": default,
        "conversations": len({r["conversation_id"] for r in rows}), "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings, "site": "resid_post at the final formatted-prompt token, before each reply",
        "warning": "Exploratory in-sample UMAP per layer. Axes, distances and paths are not refusal measurements.",
        "semantic": {"layer": default, "core_label": semantic["core_label"], "labels": semantic["counts"],
                     "radius": "refusal score: projection on mean(core) - mean(answered) of unit vectors; 0 = answered mean, 1 = core mean",
                     "angle": "atan2 of the first two principal components after removing the refusal axis",
                     "warning": "Axis defined in-sample by the displayed labels; a sketch until many turns are labelled."}
                    if semantic else {"unavailable": reason},
        "points": points,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    out.with_suffix(".json").write_text(text, encoding="utf-8")
    # A script file also opens straight from the filesystem, without a local server.
    out.with_suffix(".js").write_text(f"window.RECORDINGS={text};\n", encoding="utf-8")
    donut = (f"donut from {semantic['counts']['refused']} refused, {semantic['counts']['redirected']} redirected, "
             f"{semantic['counts']['answered']} answered" if semantic else f"no donut yet: {reason}")
    return f"{len(points)} turns from {payload['conversations']} conversations -> {out.with_suffix('.js')} ({donut})"
