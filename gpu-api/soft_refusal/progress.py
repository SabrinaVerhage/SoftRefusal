"""Private atomic progress snapshots; exposed only after job ownership checks."""
import json
import os
import time
from pathlib import Path


def write_progress(job_id, phase, **details):
    root = Path(os.environ["SOFT_REFUSAL_DATA"]) / "progress"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{job_id}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps({"job_id": job_id, "phase": phase,
                                      "updated_at": time.time(), **details}), encoding="utf-8")
    temporary.replace(target)


def cancel_marker(root, job_id):
    """The supervisor asks the separate model process to stop through this file."""
    return Path(root) / "cancel" / job_id


def request_cancel(root, job_id):
    marker = cancel_marker(root, job_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def clear_cancel(root, job_id):
    cancel_marker(root, job_id).unlink(missing_ok=True)


def read_progress(root, job_id):
    path = root / "progress" / f"{job_id}.json"
    try:
        if path.stat().st_size > 524288:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data.get("job_id") == job_id else None
    except (OSError, ValueError):
        return None
