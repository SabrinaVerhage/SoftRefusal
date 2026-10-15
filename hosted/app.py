"""One-process visitor gateway with owned sessions and a durable serial queue.

Run behind Caddy, bound to loopback. No local operator API is exposed.
"""
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
import json
import logging
import os
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from avoidance.chat import ChatService, DEFAULT_INSTRUCTION
from avoidance.io import read_json
from avoidance.remote import RemoteAdapter, load_connection

ROOT = Path(__file__).resolve().parents[1]
COOKIE = "softrefusal_visitor"
ACTIVE = ("queued", "running")
log = logging.getLogger("softrefusal")


class Gateway:
    def __init__(self, directory, connection_file, adapter_factory=None, map_enabled=True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.connection_file = Path(connection_file)
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.map_wake = threading.Event()
        self.stopping = threading.Event()
        self.accepting = True
        self.map_enabled = map_enabled
        self.health = {"online": False, "message": "Checking the model connection…"}
        self.map_state = {"status": "waiting", "message": "Waiting for the first completed conversation."}
        self.db = sqlite3.connect(self.directory / "visitors.sqlite", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
          PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS visitors(token TEXT PRIMARY KEY, created REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, owner TEXT NOT NULL, created REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, owner TEXT NOT NULL, conversation TEXT NOT NULL,
            prompt TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, error TEXT, request_key TEXT NOT NULL,
            UNIQUE(owner, request_key));
          CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created);
          CREATE TABLE IF NOT EXISTS admission(bucket TEXT NOT NULL, created REAL NOT NULL);
        """)
        self.db.execute("UPDATE jobs SET status='interrupted', error='The server restarted. Your saved replies are safe; send this message again.' WHERE status IN ('queued','running')")
        self.db.commit()
        config = read_json(ROOT / "recorder/configs/gemma3-12b-remote.json")
        factory = adapter_factory or (lambda c: RemoteAdapter(c, load_connection(self.connection_file)))
        self.service = ChatService(config, self.directory / "recordings", adapter_factory=factory,
                                   auto_label=adapter_factory is None)
        self.test_adapter = adapter_factory is not None
        self.threads = []

    def start(self):
        for target in (self.work, self.monitor, self.maps):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self.threads.append(thread)
        self.map_wake.set()

    def close(self):
        self.accepting = False
        self.stopping.set()
        self.service.stop()
        self.wake.set()
        self.map_wake.set()
        for thread in self.threads:
            thread.join(timeout=35)

    def token_owner(self, token):
        if not token or len(token) > 128:
            return None
        owner = hashlib.sha256(token.encode()).hexdigest()
        with self.lock:
            return owner if self.db.execute("SELECT 1 FROM visitors WHERE token=?", (owner,)).fetchone() else None

    def limit(self, bucket, maximum, seconds=60):
        now = time.time()
        self.db.execute("DELETE FROM admission WHERE created < ?", (now - 3600,))
        count = self.db.execute("SELECT count(*) FROM admission WHERE bucket=? AND created>?", (bucket, now - seconds)).fetchone()[0]
        if count >= maximum:
            raise HTTPException(429, "Please wait a moment before trying again.")
        self.db.execute("INSERT INTO admission VALUES (?,?)", (bucket, now))

    def create_conversation(self, token, ip):
        with self.lock:
            if not self.accepting:
                raise HTTPException(503, "The server is updating. Please try shortly.")
            self.limit("start:" + hashlib.sha256(ip.encode()).hexdigest(), 20, 3600)
            owner = self.token_owner(token)
            if owner is None:
                token = secrets.token_urlsafe(32)
                owner = hashlib.sha256(token.encode()).hexdigest()
                self.db.execute("INSERT INTO visitors VALUES (?,?)", (owner, time.time()))
            if self.db.execute("SELECT 1 FROM jobs WHERE owner=? AND status IN ('queued','running')", (owner,)).fetchone():
                raise HTTPException(409, "Wait for your reply or stop it before starting a new conversation.")
            sid = self.service.create(DEFAULT_INSTRUCTION)
            self.db.execute("INSERT INTO conversations VALUES (?,?,?)", (sid, owner, time.time()))
            self.db.commit()
            return sid, token

    def own(self, owner, sid):
        if not owner or not self.db.execute("SELECT 1 FROM conversations WHERE id=? AND owner=?", (sid, owner)).fetchone():
            raise HTTPException(404, "Conversation not found.")

    def submit(self, owner, sid, prompt, key, ip):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
            raise HTTPException(400, "Write a message of 1–4,000 characters.")
        if not isinstance(key, str) or not 8 <= len(key) <= 80:
            raise HTTPException(400, "A request key is required.")
        with self.lock:
            self.own(owner, sid)
            old = self.db.execute("SELECT * FROM jobs WHERE owner=? AND request_key=?", (owner, key)).fetchone()
            if old:
                if old["conversation"] != sid or old["prompt"] != prompt.strip():
                    raise HTTPException(409, "Request key was already used for another message.")
                return old["id"]
            if not self.accepting or not self.health["online"]:
                raise HTTPException(503, "The model is offline. Your saved conversation is still here.")
            if shutil.disk_usage(self.directory).free < 512 * 1024**2:
                raise HTTPException(503, "Recording storage is full. Please contact the installation host.")
            if self.db.execute("SELECT count(*) FROM jobs").fetchone()[0] >= 10000:
                raise HTTPException(503, "The installation has reached its recording limit.")
            if self.db.execute("SELECT 1 FROM jobs WHERE owner=? AND status IN ('queued','running')", (owner,)).fetchone():
                raise HTTPException(409, "Your previous message is still in progress.")
            if self.db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= 10:
                raise HTTPException(429, "The installation queue is full. Please try shortly.")
            self.limit("send:" + owner, 6)
            self.limit("ip:" + hashlib.sha256(ip.encode()).hexdigest(), 30)
            jid = secrets.token_hex(16)
            self.db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)", (jid, owner, sid, prompt.strip(), "queued", time.time(), None, key))
            self.db.commit()
            self.wake.set()
            return jid

    def state(self, owner, sid=None):
        with self.lock:
            if not sid and owner:
                row = self.db.execute("SELECT id FROM conversations WHERE owner=? ORDER BY created DESC LIMIT 1", (owner,)).fetchone()
                sid = row[0] if row else None
            result = {"conversation": sid, "rows": [], "job": None, "model": self.health, "map": self.map_state}
            if sid:
                self.own(owner, sid)
                state = self.service.state(sid)
                result["rows"] = [{k: row.get(k) for k in ("point_id", "turn_id", "prompt", "response", "label", "label_source", "hit_token_limit")} for row in state["rows"]]
                row = self.db.execute("SELECT * FROM jobs WHERE conversation=? ORDER BY created DESC LIMIT 1", (sid,)).fetchone()
                if row:
                    job = {k: row[k] for k in ("id", "status", "prompt", "error")}
                    if row["status"] == "queued":
                        job["position"] = self.db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running') AND created<=?", (row["created"],)).fetchone()[0]
                    if row["status"] == "running" and state["job"]:
                        job.update(response=state["job"]["response"], stage=state["job"]["stage"])
                    result["job"] = job
            return result

    def cancel(self, owner, sid):
        with self.lock:
            self.own(owner, sid)
            row = self.db.execute("SELECT * FROM jobs WHERE conversation=? AND status IN ('queued','running')", (sid,)).fetchone()
            if row and row["status"] == "queued":
                self.db.execute("UPDATE jobs SET status='stopped' WHERE id=?", (row["id"],))
                self.db.commit()
            elif row:
                self.service.stop()

    def work(self):
        while not self.stopping.is_set():
            self.wake.wait(1)
            self.wake.clear()
            with self.lock:
                row = self.db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
                if row is None:
                    continue
                try:
                    self.db.execute("UPDATE jobs SET status='running' WHERE id=?", (row["id"],))
                    self.db.commit()
                    self.service.submit(row["conversation"], row["prompt"], reply_limit=512)
                except Exception:
                    self.db.execute("UPDATE jobs SET status='error', error=? WHERE id=?", ("Could not start this reply. Please try again.", row["id"]))
                    self.db.commit()
                    continue
            deadline = time.monotonic() + 600
            while self.service.busy() and not self.stopping.wait(.25):
                if time.monotonic() > deadline:
                    self.service.stop()
            with self.lock:
                job = self.service.job or {}
                status = job.get("status", "error")
                if status == "running":
                    status = "interrupted"
                # Keep raw diagnostics in private attempt files, never visitor responses.
                error = None if status == "complete" else "Reply stopped or could not be completed. Earlier saved turns are safe. Try again or start a new conversation."
                self.db.execute("UPDATE jobs SET status=?, error=? WHERE id=?", (status, error, row["id"]))
                self.db.commit()
                if status != "complete":
                    self.service.adapter = None  # Reconnect after an offline/replaced GPU.
            self.map_wake.set()
            # Each turn uses two upstream submissions: recording and behavior tag.
            # Respect the shared GPU credential's six-submissions/minute limit.
            if not self.test_adapter:
                self.stopping.wait(21)

    def monitor(self):
        while not self.stopping.is_set():
            try:
                if self.test_adapter:
                    online = True
                else:
                    connection = load_connection(self.connection_file)
                    import urllib.request
                    req = urllib.request.Request(connection["api_url"] + "/v1/health", headers={"Authorization": "Bearer " + connection["access_code"], "User-Agent": "SoftRefusal/1"})
                    with urllib.request.urlopen(req, timeout=5) as response:
                        health = json.load(response)
                    online = health.get("status") == "accepting" and health.get("models", {}).get("gemma", {}).get("enabled") is True
                self.health = {"online": bool(online), "message": "Model online" if online else "Model paused"}
            except Exception:
                self.health = {"online": False, "message": "Model offline · saved conversations and map remain available"}
            self.stopping.wait(15)

    def maps(self):
        if not self.map_enabled:
            return
        while not self.stopping.is_set():
            self.map_wake.wait(2)
            if not self.map_wake.is_set():
                continue
            self.map_wake.clear()
            with self.lock:
                recorded = any(meta.get("latest") for meta in list(self.service.sessions.values()))
            if not recorded:
                continue
            self.map_state = {"status": "updating", "message": "Updating the conversation map…"}
            try:
                proc = subprocess.run([sys.executable, "-m", "hosted.build_map", str(self.directory)],
                                      cwd=ROOT, capture_output=True, timeout=240)
                if proc.returncode:
                    raise RuntimeError("Map worker failed")
                self.map_state = {"status": "ready", "message": "Map updated"}
            except Exception:
                log.exception("Map update failed; previous map retained")
                self.map_state = {"status": "error", "message": "Map update delayed. Saved recordings and the previous map are retained."}


def create_app(gateway=None, origin=None, secure=True):
    origin = origin or os.getenv("SOFT_REFUSAL_ORIGIN", "https://softrefusal.sabrinaverhage.com")
    @asynccontextmanager
    async def lifespan(app):
        nonlocal gateway
        if gateway is None:
            gateway = Gateway(os.getenv("SOFT_REFUSAL_DATA", "/var/lib/softrefusal"),
                              os.getenv("SOFT_REFUSAL_CONNECTION", "/etc/softrefusal/gpu-connection.json"))
        app.state.gateway = gateway
        gateway.start()
        yield
        gateway.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    from urllib.parse import urlsplit
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[urlsplit(origin).hostname, "127.0.0.1", "localhost"])

    @app.middleware("http")
    async def boundary(request, call_next):
        if request.method == "POST":
            if request.headers.get("origin") != origin or not request.headers.get("content-type", "").startswith("application/json"):
                return JSONResponse({"detail": "Request origin rejected."}, status_code=403)
            try:
                size = int(request.headers.get("content-length", "0"))
            except ValueError:
                size = 0
            if not 0 < size <= 20000:
                return JSONResponse({"detail": "Request too large or missing length."}, status_code=413)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        return response

    def owner(request):
        return gateway.token_owner(request.cookies.get(COOKIE))

    async def body_of(request):
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            raise HTTPException(400, "Invalid JSON request.")
        if not isinstance(body, dict):
            raise HTTPException(400, "Expected a JSON object.")
        return body

    @app.get("/healthz")
    def health():
        version = (ROOT / "REVISION").read_text().strip() if (ROOT / "REVISION").exists() else "development"
        return {"status": "ok", "version": version}

    @app.post("/internal/{action}")
    def maintenance(action: str, request: Request):
        if request.client.host != "127.0.0.1" or action not in {"drain", "resume"}:
            raise HTTPException(404)
        with gateway.lock:
            gateway.accepting = action == "resume"
            active = gateway.db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            return {"active": active}

    @app.get("/api/state")
    def state(request: Request, conversation: str | None = None):
        return gateway.state(owner(request), conversation)

    @app.post("/api/start")
    async def start(request: Request):
        body = await body_of(request)
        if body.get("consent") is not True:
            raise HTTPException(400, "Agree to the public recording notice before starting.")
        sid, token = gateway.create_conversation(request.cookies.get(COOKIE), request.client.host)
        response = JSONResponse({"conversation": sid})
        response.set_cookie(COOKIE, token, max_age=86400 * 7, httponly=True, secure=secure, samesite="lax")
        return response

    @app.post("/api/send")
    async def send(request: Request):
        body = await body_of(request)
        return {"job": gateway.submit(owner(request), body.get("conversation"), body.get("prompt"), body.get("request_key"), request.client.host)}

    @app.post("/api/stop")
    async def stop(request: Request):
        body = await body_of(request)
        gateway.cancel(owner(request), body.get("conversation"))
        return {"ok": True}

    @app.get("/api/map")
    def map_data():
        path = gateway.directory / "map/public.json"
        if not path.exists():
            return {"points": [], "status": gateway.map_state}
        return FileResponse(path, media_type="application/json")

    @app.get("/")
    def phone():
        return FileResponse(ROOT / "hosted/web/index.html")

    @app.get("/assets/{name}")
    def asset(name: str):
        if name not in {"phone.css", "phone.js", "live.js"}:
            raise HTTPException(404)
        return FileResponse(ROOT / "hosted/web" / name)

    @app.get("/map/")
    @app.get("/map/index.html")
    def viewer():
        text = (ROOT / "map/index.html").read_text(encoding="utf-8")
        text = text.replace('<script src="data/recordings.js"></script>', '')
        text = text.replace('</body>', '<script src="/assets/live.js"></script></body>')
        text = text.replace('href="./index.html"', 'href="/"')
        text = text.replace('Rebuild it with <code>Update Map from Recordings.cmd</code>.', 'The hosted map updates automatically after completed conversations.')
        return HTMLResponse(text)

    @app.get("/map")
    def map_redirect():
        return RedirectResponse("/map/", status_code=308)

    @app.get("/map/{name:path}")
    def map_asset(name: str):
        if name not in {"app.js", "cloud.js", "style.css", "data/sample-umap.js", "data/path-study.js"}:
            raise HTTPException(404)
        return FileResponse(ROOT / "map" / name)

    return app


app = create_app()
