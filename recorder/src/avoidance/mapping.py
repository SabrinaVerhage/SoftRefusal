"""Recorded conversations → one map file for the viewer in ../map.

Combines the latest checkpoint of every conversation in a recordings folder and
fits a separate 3D UMAP per recorded layer (cosine on raw residual vectors, the
same settings as `reduce`). Exploratory and in-sample: axes, distances and paths
are not refusal measurements.

With automatic labels or human corrections it places every turn on the "semantic donut": distance from
the centre follows a refusal axis (mean refused minus mean answered direction, as in
difference-of-means refusal-direction work), the angle follows what remains of the
state once that axis is removed. The first labelled export freezes a reference;
later exports reuse it. Reference-group mean scores are fixed by construction,
not validated predictions. The separate UMAP views remain exploratory refits."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .io import load_dataset, measurement_signature, read_json, read_jsonl
from .labels import label_records

OUTCOMES = ("answered", "redirected", "refused", "unclear", "unlabeled")


def read_labels(directory):
    """Effective automatic labels, with human corrections taking precedence."""
    return {pid: row["outcome"] for pid, row in label_records(directory).items()}


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
        current = {"layers": config["layers"], "measurement": measurement_signature(manifest, config["layers"][0]),
                   "module_path": manifest["module_path"], "site": manifest["site"]}
        if signature is None:
            signature = current
        elif current != signature:
            # 4B and 12B snapshots (or different layers) cannot share one map.
            raise ValueError(f"{directory.name} was recorded with different settings; map one model and layer set at a time")
        labels = label_records(directory)
        for row in session_rows:
            label = labels.get(row["point_id"], {})
            rows.append({**row, "conversation_number": number, "conversation_title": meta.get("title", ""),
                         "outcome": label.get("outcome", "unlabeled"), "label_source": label.get("label_source"),
                         "label_reason": label.get("reason", ""), "label_judge_version": label.get("judge_version"),
                         "label_model_id": label.get("model_id"), "label_revision": label.get("revision")})
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


def unit_vectors(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Donut requires a finite matrix of recorded vectors")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("Donut direction is undefined for zero vectors")
    return x / norms


def fit_semantic_reference(x, outcomes):
    """Fit once; the axis, PCA centre/basis and lift scale are reused unchanged."""
    outcomes = np.asarray(outcomes)
    core, outer = outcomes == "refused", outcomes == "answered"
    if not core.any() or not outer.any():
        return None, "Label at least one reply as refused and one as answered. Redirection alone does not define a refusal reference."
    u = unit_vectors(x)
    axis = u[core].mean(0) - u[outer].mean(0)
    length = np.linalg.norm(axis)
    if length <= 1e-10:
        return None, "The answered and refused reference states have no distinct direction. Add varied labelled recordings."
    axis /= length
    raw = u @ axis
    low, high = raw[outer].mean(), raw[core].mean()
    rest = u - np.outer(raw, axis)
    center = rest.mean(0)
    _, singular, vt = np.linalg.svd(rest - center, full_matrices=False)
    basis = np.zeros((3, x.shape[1]))
    for i in range(min(3, len(vt))):
        if singular[i] > 1e-10:
            basis[i] = vt[i]
            # Resolve the arbitrary PCA sign when creating a new reference.
            if basis[i, np.argmax(np.abs(basis[i]))] < 0:
                basis[i] *= -1
    pcs = (rest - center) @ basis.T
    return {"axis": axis.tolist(), "low": float(low), "high": float(high),
            "center": center.tolist(), "basis": basis.tolist(),
            "lift_scale": float(np.abs(pcs[:, 2]).max() or 1), "dimensions": x.shape[1],
            "counts": {k: int((outcomes == k).sum()) for k in OUTCOMES}}, None


def semantic_layout(x, outcomes, reference=None):
    if reference is None:
        reference, reason = fit_semantic_reference(x, outcomes)
        if reference is None:
            return None, reason
    u = unit_vectors(x)
    axis, basis, center = (np.asarray(reference[k]) for k in ("axis", "basis", "center"))
    if (axis.shape != (u.shape[1],) or center.shape != axis.shape or basis.shape != (3, u.shape[1])
            or not all(np.isfinite(a).all() for a in (axis, basis, center))
            or not np.isfinite([reference["low"], reference["high"], reference["lift_scale"]]).all()
            or reference["high"] - reference["low"] <= 1e-10 or reference["lift_scale"] <= 0):
        raise ValueError("Invalid donut reference; restore its saved copy or explicitly rebuild it")
    raw = u @ axis
    pcs = (u - np.outer(raw, axis) - center) @ basis.T
    outcomes = np.asarray(outcomes)
    return {"score": (raw - reference["low"]) / (reference["high"] - reference["low"]),
            "angle": np.arctan2(pcs[:, 1], pcs[:, 0]),
            "lift": np.clip(pcs[:, 2] / reference["lift_scale"], -1, 1), "core_label": "refused",
            "counts": {k: int((outcomes == k).sum()) for k in OUTCOMES}}, None


def export_map(runs, out, layer=None, seed=42, neighbors=10, min_dist=0.1, rebuild_reference=False, allow_small=False):
    rows, matrices, signature = collect_recordings(runs)
    layers = list(signature["layers"])
    model_id, revision = (signature["measurement"][k] for k in ("model_id", "revision"))
    default = layer if layer is not None else layers[len(layers) // 2 - 1 if len(layers) > 1 else 0]
    if default not in layers:
        raise ValueError(f"Layer {default} was not recorded; choose one of {layers}")
    out = Path(out)
    reference_path = out.with_suffix(".reference.json")
    reference = read_json(reference_path) if reference_path.exists() and not rebuild_reference else None
    expected = {**signature, "donut_layer": default, "dimensions": matrices[f"layer_{default}"].shape[1]}
    if reference is not None and (reference.get("version") != 1 or reference.get("signature") != expected):
        raise ValueError("Saved donut reference has different measurement settings. Use another output path or --rebuild-reference.")
    outcomes = [r["outcome"] for r in rows]
    new_reference = reference is None
    reason = None
    if new_reference:
        geometry, reason = fit_semantic_reference(matrices[f"layer_{default}"], outcomes)
        if geometry:
            reference = {"version": 1, "signature": expected, "geometry": geometry,
                         "created_utc": datetime.now(timezone.utc).isoformat(),
                         "anchors": [{k: r.get(k) for k in ("point_id", "outcome", "label_source", "label_judge_version", "label_model_id", "label_revision")} for r in rows],
                         "activation_sha256": hashlib.sha256(matrices[f"layer_{default}"].tobytes()).hexdigest()}
            reference["id"] = hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()[:16]
        elif rebuild_reference:
            raise ValueError(f"Cannot rebuild reference: {reason} Previous files were kept.")
    semantic = semantic_layout(matrices[f"layer_{default}"], outcomes, reference["geometry"])[0] if reference else None
    coordinates, settings = {}, None
    for n in layers:
        matrix = matrices[f"layer_{n}"]
        if allow_small and (len(matrix) < 6 or len(np.unique(matrix, axis=0)) < 6):
            centered = matrix.astype(np.float64) - matrix.mean(axis=0)
            u, singular, _ = np.linalg.svd(centered, full_matrices=False)
            coords = np.zeros((len(matrix), 3))
            count = min(3, len(singular))
            coords[:, :count] = u[:, :count] * singular[:count]
            coordinates[n] = coords
            settings = {"method": "pca_preview", "n_components": 3}
        else:
            coordinates[n], settings = fit_layer(matrix, seed, neighbors, min_dist)
            settings["method"] = "umap"
    previous, points = {}, []
    for i, row in enumerate(rows):
        cid = row["conversation_id"]
        points.append({
            "point_id": row["point_id"], "conversation_id": cid, "turn_id": row["turn_id"],
            "previous_point_id": previous.get(cid),
            "completed_utc": row.get("completed_utc"),
            "conversation_number": row["conversation_number"], "conversation_title": row["conversation_title"],
            "prompt": row["prompt"], "response": row["response"], "outcome": row.get("outcome", "unlabeled"),
            "label_source": row.get("label_source"), "label_reason": row.get("label_reason", ""),
            "label_judge_version": row.get("label_judge_version"),
            "label_model_id": row.get("label_model_id"), "label_revision": row.get("label_revision"),
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
        "warning": ("Early PCA preview, not UMAP: waiting for six distinct recorded turns. "
                    if settings.get("method") == "pca_preview" else "Exploratory in-sample UMAP per layer. ")
                   + "Axes, distances and paths are not refusal measurements.",
        "semantic": {"layer": default, "core_label": semantic["core_label"], "labels": semantic["counts"],
                     "radial_mapping": "smooth_v1",
                     "reference_id": reference["id"], "reference_created_utc": reference["created_utc"],
                     "reference_labels": reference["geometry"]["counts"], "frozen": True,
                     "reference_label_sources": {source: sum(a.get("label_source") == source and a["outcome"] in ("answered", "refused") for a in reference["anchors"])
                                                 for source in ("human", "model_judge")},
                     "radius": "refusal score: projection on mean(core) - mean(answered) of unit vectors; 0 = answered mean, 1 = core mean",
                     "angle": "atan2 of the first two principal components after removing the refusal axis",
                     "warning": "Fixed exploratory reference, not a calibrated probability or causal boundary. Anchors may use fallible automatic judgments and were fitted in-sample. UMAP views still refit."}
                    if semantic else {"unavailable": reason},
        "points": points,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if reference and new_reference:
        if reference_path.exists():
            old = read_json(reference_path)
            reference_path.with_name(f"{out.stem}.reference-{old['id']}.json").write_bytes(reference_path.read_bytes())
        temporary = reference_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(reference, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(reference_path)
    out.with_suffix(".json").write_text(text, encoding="utf-8")
    # A script file also opens straight from the filesystem, without a local server.
    out.with_suffix(".js").write_text(f"window.RECORDINGS={text};\n", encoding="utf-8")
    donut = (f"donut from {semantic['counts']['refused']} refused, {semantic['counts']['redirected']} redirected, "
             f"{semantic['counts']['answered']} answered" if semantic else f"no donut yet: {reason}")
    return f"{len(points)} turns from {payload['conversations']} conversations -> {out.with_suffix('.js')} ({donut})"
