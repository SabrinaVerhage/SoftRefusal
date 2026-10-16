"""Build an allowlisted deployment ZIP; credentials/caches are never included."""
import hashlib
import zipfile
from pathlib import Path


root = Path(__file__).resolve().parents[1]
target = root / "soft-refusal-deploy.zip"
files = [root / name for name in ("AGENTS.md", "README.md", "DEPLOYMENT_LOG.md", "DEPLOYMENT_STATUS.md", "requirements-api.txt", "requirements-gpu.txt", ".env.example", ".gitignore")]
files.append(root / "requirements-lora.txt")
files.append(root / "LORAS.md")
for folder, suffixes in (("soft_refusal", {".py", ".json"}), ("public", {".html", ".css", ".js"}),
                          ("scripts", {".py", ".sh"}), ("tests", {".py"})):
    files.extend(p for p in (root / folder).rglob("*") if p.is_file() and p.suffix in suffixes)
with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, "soft-refusal/" + path.relative_to(root).as_posix())
with zipfile.ZipFile(target) as archive:
    assert not any("secrets/" in n or ".venv/" in n or "runtime/" in n for n in archive.namelist())
print(f"Prepared {len(files)} source files: {target}")
print("SHA256:", hashlib.sha256(target.read_bytes()).hexdigest())
