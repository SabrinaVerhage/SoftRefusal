"""Rebuild the synthetic-vector UMAP sample with the Recording Room environment.

This is a real UMAP reduction of synthetic vectors, not recorded model behavior.
Only centering, uniform scaling, and a rigid rotation are applied for display.
"""
from pathlib import Path
import json
import numpy as np
import umap

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)
n = 3900
theta = rng.uniform(0, 2 * np.pi, n)
phi = rng.uniform(0, 2 * np.pi, n)
radius = 2.1 + 0.66 * np.cos(phi)
latent = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), 0.66 * np.sin(phi)))
# A linear lift into 48 dimensions preserves a known synthetic manifold.
basis, _ = np.linalg.qr(rng.normal(size=(48, 3)))
vectors = latent @ basis.T + rng.normal(0, 0.016, (n, 48))
settings = dict(n_components=3, n_neighbors=36, min_dist=0.3, metric="euclidean", random_state=42, n_jobs=1)
coordinates = umap.UMAP(**settings).fit_transform(vectors)
centered = coordinates - coordinates.mean(axis=0)
_, _, rotation = np.linalg.svd(centered, full_matrices=False)
display = centered @ rotation.T
display *= 2.8 / np.percentile(np.linalg.norm(display, axis=1), 97)
topics = ["consent", "intimacy", "desire", "erotic fiction", "relationships", "sexual health"]
prompts = [
    "How can I talk about consent with a new partner?",
    "How do I talk about intimacy without feeling ashamed?",
    "Why is it difficult to put desire into words?",
    "Can you help me write a romantic scene?",
    "How can I tell my partner what I need?",
    "Where can I find clear information about sexual health?",
]
points = []
for i, (raw, xyz, angle, tube) in enumerate(zip(coordinates, display, theta, phi)):
    topic = int(angle / (2 * np.pi) * 6) % 6
    # Outcomes are explicitly assigned for the demo; no model is queried.
    outcome = "refused" if topic == 3 and np.cos(tube) < 0.4 else "redirected" if np.sin(tube) > 0.72 else "answered"
    points.append(dict(point_id=f"sample-{i:04d}", x=round(float(xyz[0]), 6), y=round(float(xyz[1]), 6), z=round(float(xyz[2]), 6), original_coordinates=[round(float(x), 6) for x in raw], topic=topics[topic], prompt=prompts[topic], outcome=outcome, synthetic=True))
payload = dict(name="Synthetic-vector UMAP", kind="umap", synthetic=True, settings=settings, source="3,900 synthetic 48-dimensional vectors; outcomes assigned for demonstration", display_transform="Mean centered, rigid PCA rotation, uniform scale. No nonlinear warping.", points=points)
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "data" / "sample-umap.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
(ROOT / "data" / "sample-umap.js").write_text("window.UMAP_SAMPLE=" + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
print(f"Saved {len(points):,} points; umap-learn {umap.__version__}")
