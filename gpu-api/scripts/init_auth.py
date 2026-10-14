"""Generate app-only credentials into private local files. Print paths, never tokens."""
import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("secrets"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    auth_file = args.directory / "auth.json"
    if auth_file.exists():
        raise SystemExit("Existing credentials preserved. Use the current files or rotate explicitly.")
    rows = []
    for subject in ("operator", "rehearsal-1", "rehearsal-2"):
        token = secrets.token_urlsafe(32)
        path = args.directory / f"{subject}-token.txt"
        # Exclusive creation avoids overwriting a retained access code.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(token + "\n")
        os.chmod(path, 0o600)
        rows.append({"subject": subject, "sha256": hashlib.sha256(token.encode()).hexdigest(),
                     "admin": subject == "operator"})
    with auth_file.open("x", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
    os.chmod(auth_file, 0o600)
    print(f"App credentials created in {args.directory.resolve()}; tokens were not printed.")
    print("Use an application access code in the prototype. Infrastructure credentials never belong there.")


if __name__ == "__main__":
    main()
