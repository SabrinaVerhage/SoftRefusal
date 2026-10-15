"""Build privately, then atomically publish only the viewer payload."""
import json
import os
from pathlib import Path
import sys

from avoidance.mapping import export_map


def build(directory):
    directory = Path(directory)
    out = directory / "map"
    out.mkdir(exist_ok=True)
    export_map(directory / "recordings", out / "recordings.js", allow_small=True)
    payload = json.loads((out / "recordings.json").read_text(encoding="utf-8"))
    # The automatic judge's raw evidence and internal diagnostics remain private.
    for point in payload["points"]:
        point.pop("label_reason", None)
    temporary = out / "public.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, out / "public.json")


if __name__ == "__main__":
    build(sys.argv[1])
