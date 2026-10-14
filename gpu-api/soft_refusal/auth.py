import hashlib
import hmac
import json
from dataclasses import dataclass

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class Principal:
    subject: str
    admin: bool = False


class Auth:
    def __init__(self, path):
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ValueError("Configure at least one application credential")
        self.records = []
        subjects = set()
        for row in records:
            digest, subject = row.get("sha256", ""), row.get("subject", "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Invalid application credential digest")
            if not subject or subject in subjects:
                raise ValueError("Application credential subjects must be unique")
            subjects.add(subject)
            self.records.append((digest, Principal(subject, bool(row.get("admin", False)))))

    async def __call__(self, request: Request):
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer ") or not 32 <= len(header[7:]) <= 256:
            raise HTTPException(401, detail={"code": "unauthorized"}, headers={"WWW-Authenticate": "Bearer"})
        supplied = hashlib.sha256(header[7:].encode()).hexdigest()
        for digest, principal in self.records:
            if hmac.compare_digest(digest, supplied):
                return principal
        raise HTTPException(401, detail={"code": "unauthorized"}, headers={"WWW-Authenticate": "Bearer"})
