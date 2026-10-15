"""Reply-behaviour annotations, separate from immutable activation checkpoints."""
from pathlib import Path

from .io import read_jsonl

OUTCOMES = ("answered", "redirected", "refused", "unclear", "unlabeled")


def label_records(directory):
    """Human corrections always win, regardless of when an automatic job finishes."""
    directory = Path(directory)
    automatic, human = {}, {}
    for name, target in (("auto_labels.jsonl", automatic), ("labels.jsonl", human)):
        path = directory / name
        if not path.exists():
            continue
        for row in read_jsonl(path):
            pid = row["point_id"]
            if name == "labels.jsonl" and row.get("outcome") == "auto":
                target.pop(pid, None)
            elif row.get("outcome") in OUTCOMES:
                target[pid] = {**row, "label_source": "human" if name == "labels.jsonl" else "model_judge"}
    return {**automatic, **human}
