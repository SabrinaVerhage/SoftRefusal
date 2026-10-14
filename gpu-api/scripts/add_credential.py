"""Add one app-only credential to an existing auth file. Prints paths, never tokens.

The API reads auth.json at startup: copy the updated file to the Pod's
/workspace/soft-refusal/secrets/ and restart the API for it to take effect."""
import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", help="Name for this client, e.g. recording-room")
    parser.add_argument("--directory", type=Path, default=Path("secrets"))
    args = parser.parse_args()
    auth_file = args.directory / "auth.json"
    rows = json.loads(auth_file.read_text(encoding="utf-8"))
    if any(row["subject"] == args.subject for row in rows):
        raise SystemExit(f"'{args.subject}' already exists. Rotate it explicitly instead.")
    token = secrets.token_urlsafe(32)
    path = args.directory / f"{args.subject}-token.txt"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(token + "\n")
    os.chmod(path, 0o600)
    rows.append({"subject": args.subject, "sha256": hashlib.sha256(token.encode()).hexdigest(), "admin": False})
    temporary = auth_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    temporary.replace(auth_file)
    print(f"Added '{args.subject}'. Access code saved to {path.resolve()}; it was not printed.")


if __name__ == "__main__":
    main()
