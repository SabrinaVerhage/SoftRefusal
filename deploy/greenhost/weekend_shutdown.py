#!/usr/bin/env python3
"""Stop only the approved weekend Pod, using its existing scoped runtime key.

Installed on the existing VPS as a root-only systemd timer. --check is read-only.
No credentials or raw API responses are logged.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import urllib.request


def request(key, query):
    req = urllib.request.Request("https://api.runpod.io/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "User-Agent": "SoftRefusalShutdown/1"})
    with urllib.request.urlopen(req, timeout=20) as response:
        result = json.load(response)
    if result.get("errors") or not result.get("data"):
        raise RuntimeError("Provider rejected shutdown API request")
    return result["data"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--config", default="/etc/softrefusal/weekend-stop.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    pod = config["pod_id"]
    if not re.fullmatch(r"[a-z0-9]{8,32}", pod):
        raise ValueError("Invalid configured Pod ID")
    deadline = datetime.fromisoformat(config["deadline_utc"])
    if deadline.tzinfo is None:
        raise ValueError("Shutdown deadline must include a timezone")
    key = Path(config["key_file"]).read_text().strip()
    current = request(key, 'query { pod(input: {podId: "' + pod + '"}) { id desiredStatus } }')["pod"]
    if not current or current.get("id") != pod:
        raise RuntimeError("Configured Pod could not be verified")
    if args.check:
        print(json.dumps({"check": "read_only", "pod_id": pod,
                          "status": current["desiredStatus"], "deadline_utc": deadline.isoformat()}))
        return
    if datetime.now(timezone.utc) < deadline:
        raise RuntimeError("Refusing to stop before the approved deadline")
    if current["desiredStatus"] == "EXITED":
        print("Approved weekend Pod is already stopped.")
        return
    result = request(key, 'mutation { podStop(input: {podId: "' + pod + '"}) { id desiredStatus } }')["podStop"]
    if result.get("id") != pod or result.get("desiredStatus") != "EXITED":
        raise RuntimeError("Provider did not confirm the requested stop")
    print("Provider accepted stop for the approved weekend Pod. Network volume retained.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Weekend shutdown check/action failed: " + type(exc).__name__, file=sys.stderr)
        sys.exit(1)
