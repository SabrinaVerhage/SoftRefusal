from __future__ import annotations

import numpy as np

from .io import digest, environment, load_dataset, new_directory, write_json, write_jsonl


def reduce(run, output, layer, seed=42, neighbors=10, min_dist=0.1):
    import joblib
    from sklearn.manifold import trustworthiness
    from umap import UMAP
    manifest, rows, arrays = load_dataset(run)
    x = arrays[f"layer_{layer}"]
    if len(x) < 6 or len(np.unique(x, axis=0)) < 6:
        raise ValueError("Need at least six distinct points for this exploratory 3D UMAP")
    if (np.linalg.norm(x, axis=1) == 0).any():
        raise ValueError("Cosine distance is undefined for zero vectors")
    if neighbors < 2 or min_dist < 0 or min_dist > 1:
        raise ValueError("Require neighbors >= 2 and 0 <= min_dist <= 1")
    out = new_directory(output)
    settings = dict(n_components=3, n_neighbors=min(neighbors, len(x) - 1), min_dist=min_dist,
                    metric="cosine", init="random", random_state=seed, transform_seed=seed, n_jobs=1)
    reducer = UMAP(**settings)
    coords = reducer.fit_transform(x)
    if not np.isfinite(coords).all():
        raise ValueError("UMAP produced non-finite coordinates")
    previous = {}
    points = []
    for row, xyz in zip(rows, coords):
        cid = row["conversation_id"]
        points.append({"point_id": row["point_id"], "conversation_id": cid, "turn_id": row["turn_id"],
                       "previous_point_id": previous.get(cid), "x": float(xyz[0]), "y": float(xyz[1]),
                       "z": float(xyz[2]), "outcome": row["outcome"],
                       "coordinate_kind": "UMAP of residual activations", "synthetic": manifest["synthetic"]})
        previous[cid] = row["point_id"]
    write_jsonl(out / "points.jsonl", points)
    np.save(out / "coordinates.npy", coords)
    joblib.dump(reducer, out / "umap.joblib")
    k = min(5, (len(x) - 1) // 2)
    write_json(out / "reduction.json", {"source_manifest_sha256": digest(str(run) + "/manifest.json"),
               "layer": layer, "settings": settings, "preprocessing": "none; cosine on raw residual vectors",
               "environment": environment(), "synthetic": manifest["synthetic"],
               "trustworthiness_k": k,
               "trustworthiness": float(trustworthiness(x, coords, n_neighbors=k, metric="cosine")),
               "warning": "Exploratory in-sample layout. Axes, distance and trajectory lines are not refusal measurements."})
    return out
