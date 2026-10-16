"""Owned image jobs on the existing durable visitor queue; credentials stay private."""
import hashlib
import json
import re
import secrets
import shutil
import time
import urllib.error
import urllib.request

from fastapi import HTTPException
from avoidance.remote import load_connection


class ImageJobs:
    def __init__(self, gateway):
        self.g = gateway
        self.online = False
        self.progress = {}

    def request(self, path, body=None, key=None, binary=False):
        c = load_connection(self.g.connection_file)
        headers = {"Authorization": "Bearer " + c["access_code"], "User-Agent": "SoftRefusal/1"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if key:
            headers["Idempotency-Key"] = key
        req = urllib.request.Request(c["api_url"] + path,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=20) as response:
            data = response.read(12 * 1024**2 + 1)
            if len(data) > 12 * 1024**2:
                raise ValueError("Oversized upstream result")
            if binary:
                if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("Expected a PNG result")
                return data
            return json.loads(data)

    def session(self, token, ip):
        g = self.g
        with g.lock:
            if not g.accepting:
                raise HTTPException(503, "The server is updating. Please try shortly.")
            g.limit("image-session:" + hashlib.sha256(ip.encode()).hexdigest(), 20, 3600)
            if not g.token_owner(token):
                token = secrets.token_urlsafe(32)
                g.db.execute("INSERT INTO visitors VALUES (?,?)", (hashlib.sha256(token.encode()).hexdigest(), time.time()))
            g.db.commit()
            return token

    def submit(self, owner, body, ip):
        if not owner:
            raise HTTPException(401, "Enter the image testing room first.")
        prompt, key = body.get("prompt"), body.get("request_key")
        width, height, seed = (body.get(k, d) for k, d in (("width", 512), ("height", 512), ("seed", 42)))
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
            raise HTTPException(400, "Write a prompt of 1–4,000 characters.")
        if not isinstance(key, str) or not 8 <= len(key) <= 80:
            raise HTTPException(400, "A request key is required.")
        if any(type(v) is not int or v not in (512, 768, 1024) for v in (width, height)):
            raise HTTPException(400, "Choose 512, 768 or 1024 pixels.")
        if type(seed) is not int or not 0 <= seed <= 2147483647:
            raise HTTPException(400, "Seed must be an integer between 0 and 2147483647.")
        payload = json.dumps(dict(model="flux", prompt=prompt.strip(), width=width, height=height, seed=seed, steps=4), sort_keys=True)
        g = self.g
        with g.lock:
            old = g.db.execute("SELECT * FROM jobs WHERE owner=? AND request_key=?", (owner, key)).fetchone()
            if old:
                if old["kind"] != "image" or old["payload"] != payload:
                    raise HTTPException(409, "Request key was already used for a different request.")
                return old["id"]
            if not g.accepting or not self.online:
                raise HTTPException(503, "The image model is offline. Try again when it is online.")
            if g.db.execute("SELECT 1 FROM jobs WHERE owner=? AND status IN ('queued','running')", (owner,)).fetchone():
                raise HTTPException(409, "Your previous image or chat request is still in progress.")
            if g.db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= 10:
                raise HTTPException(429, "The shared queue is full. Try shortly.")
            if g.db.execute("SELECT count(*) FROM jobs").fetchone()[0] >= 10000:
                raise HTTPException(503, "The installation has reached its recording limit.")
            if shutil.disk_usage(g.directory).free < 512 * 1024**2:
                raise HTTPException(503, "Image storage is full. Contact the installation host.")
            g.limit("send:" + owner, 6)
            g.limit("ip:" + hashlib.sha256(ip.encode()).hexdigest(), 30)
            jid = secrets.token_hex(16)
            g.db.execute("INSERT INTO jobs(id,owner,conversation,prompt,status,created,request_key,kind,payload) VALUES (?,?,'',?,'queued',?,?,'image',?)",
                         (jid, owner, prompt.strip(), time.time(), key, payload))
            g.db.commit()
            g.wake.set()
            return jid

    def state(self, owner):
        g = self.g
        with g.lock:
            jobs = []
            for row in g.db.execute("SELECT * FROM jobs WHERE owner=? AND kind='image' ORDER BY created DESC LIMIT 20", (owner,)):
                item = {k: row[k] for k in ("id", "prompt", "status", "error")}
                item["settings"] = json.loads(row["payload"])
                if row["status"] == "queued":
                    item["position"] = g.db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running') AND created<=?", (row["created"],)).fetchone()[0]
                item.update(self.progress.get(row["id"], {}))
                if row["status"] == "complete":
                    item["result_url"] = f"/api/images/jobs/{row['id']}/result"
                jobs.append(item)
            active = bool(g.db.execute("SELECT 1 FROM jobs WHERE owner=? AND status IN ('queued','running')", (owner,)).fetchone())
            return {"signed_in": bool(owner), "online": self.online, "busy": active, "jobs": jobs,
                    "model": "FLUX.2 klein 4B", "precision": "bfloat16", "steps": 4}

    def run(self, row):
        g, jid = self.g, row["id"]
        deadline = time.monotonic() + 600
        upstream = None
        try:
            # The same key is retained across ambiguous network failures and rate limits.
            while not g.stopping.is_set():
                try:
                    if upstream is None:
                        result = self.request("/v1/images/jobs", json.loads(row["payload"]), "hosted-image-" + jid)
                        upstream = result["job_id"]
                        if not isinstance(upstream, str) or not re.fullmatch(r"[a-f0-9]{32}", upstream):
                            raise ValueError("Invalid upstream job ID")
                        with g.lock:
                            g.db.execute("UPDATE jobs SET upstream_id=? WHERE id=?", (upstream, jid))
                            g.db.commit()
                    else:
                        result = self.request("/v1/jobs/" + upstream)
                    status = result["status"]
                    progress = result.get("progress") or {}
                    stage = progress.get("stage")
                    if stage not in {"loading", "preparing", "generating"}:
                        stage = "working"
                    with g.lock:
                        self.progress[jid] = {"stage": stage, "upstream_status": status,
                                              "upstream_position": result.get("queue_position")}
                    if status == "succeeded":
                        available = (result.get("result") or {}).get("image_available")
                        if not available:
                            self.finish(jid, "no_image", "No image was returned. This is not automatically classified as a refusal.")
                        else:
                            data = self.request("/v1/jobs/" + upstream + "/result", binary=True)
                            folder = g.directory / "images"
                            folder.mkdir(exist_ok=True)
                            temporary = folder / (jid + ".tmp")
                            temporary.write_bytes(data)
                            temporary.replace(folder / (jid + ".png"))
                            self.finish(jid, "complete", None)
                        return
                    if status not in {"queued", "running"}:
                        code = ((result.get("result") or {}).get("error") or {}).get("code", "generation_failed")
                        if not isinstance(code, str) or not re.fullmatch(r"[a-z_]{1,60}", code):
                            code = "generation_failed"
                        self.finish(jid, "error", "Technical error: " + code + ". This is not a refusal label.")
                        return
                except urllib.error.HTTPError as exc:
                    if exc.code not in {429, 502, 503, 504}:
                        raise
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Image request timed out")
                g.stopping.wait(2)
            self.finish(jid, "interrupted", "The website restarted. The GPU may still finish this request; it will not be submitted again automatically.")
        except Exception:
            self.finish(jid, "error", "Technical error: the image request could not be completed. This is not a refusal label.")

    def finish(self, jid, status, error):
        with self.g.lock:
            self.g.db.execute("UPDATE jobs SET status=?,error=? WHERE id=?", (status, error, jid))
            self.g.db.commit()
            self.progress.pop(jid, None)

    def result(self, owner, jid):
        with self.g.lock:
            row = self.g.db.execute("SELECT * FROM jobs WHERE id=? AND owner=? AND kind='image'", (jid, owner)).fetchone()
            if not row:
                raise HTTPException(404, "Image not found.")
            if row["status"] != "complete":
                raise HTTPException(409, "This job has no completed image.")
            upstream = row["upstream_id"]
        path = self.g.directory / "images" / (row["id"] + ".png")
        if path.is_file():
            return path.read_bytes()
        try:
            return self.request("/v1/jobs/" + upstream + "/result", binary=True)
        except Exception:
            raise HTTPException(503, "Image unavailable while the GPU service is offline. The job record is retained.")
