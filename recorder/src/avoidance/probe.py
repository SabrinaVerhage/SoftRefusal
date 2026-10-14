from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .io import digest, environment, load_dataset, measurement_signature, new_directory, write_json, write_jsonl

OUTCOMES = {"allow", "refusal", "mixed", "unclear", "unlabeled"}


def read_annotations(path, rows):
    with open(path, newline="", encoding="utf-8-sig") as f:
        annotations = list(csv.DictReader(f))
    by_id = {a["point_id"]: a for a in annotations}
    if len(by_id) != len(annotations) or set(by_id) != {r["point_id"] for r in rows}:
        raise ValueError("Annotation IDs must match every dataset point exactly once")
    groups, selected = {}, []
    for index, row in enumerate(rows):
        a = by_id[row["point_id"]]
        if a["outcome"] not in OUTCOMES:
            raise ValueError("Unknown annotation outcome")
        if a["outcome"] in {"unlabeled", "mixed", "unclear"}:
            continue
        if not a["annotator"].strip() or a["split"] not in {"train", "calibration", "test"}:
            raise ValueError("Each binary label requires annotator and an explicit split")
        for key in [("family", row["group_id"]), ("conversation", row["conversation_id"])]:
            if key in groups and groups[key] != a["split"]:
                raise ValueError("Data leakage: related conversations/turns cross splits")
            groups[key] = a["split"]
        selected.append((index, a, row))
    for split, minimum in [("train", 5), ("calibration", 3), ("test", 3)]:
        for outcome in ["allow", "refusal"]:
            gs = {r["group_id"] for _, a, r in selected if a["split"] == split and a["outcome"] == outcome}
            if len(gs) < minimum:
                raise ValueError(f"Need >= {minimum} independent groups with {outcome} labels in {split}; neutral smoke data cannot train a refusal probe")
    return selected


def metrics(y, p, weights=None):
    return {"roc_auc": float(roc_auc_score(y, p, sample_weight=weights)),
            "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, p >= .5, sample_weight=weights)),
            "brier": float(brier_score_loss(y, p, sample_weight=weights)),
            "log_loss": float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
            "confusion_matrix_allow_refusal": confusion_matrix(y, p >= .5, sample_weight=weights, labels=[0, 1]).tolist()}


def group_weights(groups):
    return np.array([1 / np.sum(groups == g) for g in groups])


def train_probe(run, annotations, output, layer=17, seed=42):
    manifest, rows, arrays = load_dataset(run)
    if manifest["synthetic"]:
        raise ValueError("Synthetic data cannot train a research refusal probe")
    if manifest["config"]["token_selection"] != "last_prompt":
        raise ValueError("Predictive probe requires pre-response activations; response text would leak the outcome")
    selected = read_annotations(annotations, rows)
    x = arrays[f"layer_{layer}"]
    partitions = {}
    for split in ["train", "calibration", "test"]:
        part = [(i, a, r) for i, a, r in selected if a["split"] == split]
        partitions[split] = (np.array([i for i, _, _ in part]),
                             np.array([int(a["outcome"] == "refusal") for _, a, _ in part]),
                             np.array([r["group_id"] for _, _, r in part]))
    train_i, train_y, train_g = partitions["train"]
    cal_i, cal_y, cal_g = partitions["calibration"]
    test_i, test_y, test_g = partitions["test"]
    base = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=seed))
    base.fit(x[train_i], train_y, logisticregression__sample_weight=group_weights(train_g))
    # Fixed regularized Platt calibration; calibration labels never fit the feature direction.
    calibrator = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
    calibrator.fit(base.decision_function(x[cal_i]).reshape(-1, 1), cal_y, sample_weight=group_weights(cal_g))
    p = calibrator.predict_proba(base.decision_function(x[test_i]).reshape(-1, 1))[:, 1]
    report = metrics(test_y, p, group_weights(test_g))
    rng = np.random.default_rng(seed)
    unique = np.unique(test_g)
    boot = []
    for _ in range(500):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(test_g == group) for group in sampled])
        w = np.concatenate([np.full(np.sum(test_g == group), 1 / np.sum(test_g == group)) for group in sampled])
        if len(np.unique(test_y[idx])) == 2:
            boot.append(roc_auc_score(test_y[idx], p[idx], sample_weight=w))
    out = new_directory(output)
    report.update({"test_auc_group_bootstrap_95_percentile": np.quantile(boot, [.025, .975]).tolist() if boot else None,
                   "bootstrap_valid_replicates": len(boot), "metric_weighting": "equal total weight per group",
                   "split_counts": {s: {"points": len(v[0]), "groups": len(np.unique(v[2]))} for s, v in partitions.items()},
                   "majority_baseline_balanced_accuracy": 0.5,
                   "calibration_prevalence_baseline_brier": float(brier_score_loss(test_y,
                      np.full(len(test_y), np.average(cal_y, weights=group_weights(cal_g))), sample_weight=group_weights(test_g))),
                   "settings": {"C": 1.0, "seed": seed, "scaler": "train-only StandardScaler", "threshold": 0.5},
                   "source_manifest_sha256": digest(Path(run) / "manifest.json"),
                   "annotations_sha256": digest(annotations), "environment": environment(),
                   "warning": "Association with human refusal labels, not causality or a universal sexual-avoidance axis; small group counts give unstable estimates."})
    joblib.dump({"base": base, "calibrator": calibrator, "signature": measurement_signature(manifest, layer)}, out / "probe.joblib")
    write_json(out / "evaluation.json", report)
    write_jsonl(out / "heldout_scores.jsonl", [
        {"point_id": rows[i]["point_id"], "outcome": "refusal" if y else "allow",
         "refusal_score": float(value), "split": "test"} for i, y, value in zip(test_i, test_y, p)])
    # Keep the exact annotation evidence beside the fitted probe.
    (out / "annotations.csv").write_bytes(Path(annotations).read_bytes())
    return out


def score(run, probe_path, output, layer=17):
    manifest, rows, arrays = load_dataset(run)
    bundle = joblib.load(probe_path)  # Only load artifacts produced by this project / trusted collaborators.
    if bundle["signature"] != measurement_signature(manifest, layer):
        raise ValueError("Probe/model/revision/dtype/layer/token-site signature mismatch")
    x = arrays[f"layer_{layer}"]
    raw = bundle["base"].decision_function(x)
    p = bundle["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]
    out = new_directory(output)
    write_jsonl(out / "scores.jsonl", [{"point_id": r["point_id"], "refusal_score": float(v),
                 "uncalibrated_logit": float(z), "interpretation": "probe estimate, not human outcome"}
                 for r, v, z in zip(rows, p, raw)])
    write_json(out / "scores.json", {"source_manifest_sha256": digest(Path(run) / "manifest.json"),
               "probe_sha256": digest(probe_path), "signature": bundle["signature"],
               "warning": "Scores on training/calibration points are not held-out performance."})
    return out
