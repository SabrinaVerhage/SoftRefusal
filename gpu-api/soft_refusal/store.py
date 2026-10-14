import json
import sqlite3
import time
import uuid


class Store:
    """Single supervisor owns this connection; records survive Pod restarts."""

    def __init__(self, root):
        root.mkdir(parents=True, exist_ok=True)
        (root / "images").mkdir(exist_ok=True)
        (root / "activations").mkdir(exist_ok=True)
        self.db = sqlite3.connect(root / "jobs.sqlite")
        self.db.row_factory = sqlite3.Row
        # DELETE journaling avoids WAL locking assumptions on network volumes.
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL,
            model TEXT NOT NULL, request TEXT NOT NULL, status TEXT NOT NULL,
            created REAL NOT NULL, started REAL, finished REAL, result TEXT,
            idempotency_key TEXT, request_hash TEXT,
            UNIQUE(owner, idempotency_key))""")
        self.db.execute("""UPDATE jobs SET status='interrupted', finished=?, result=?
            WHERE status IN ('queued','running')""", (time.time(), json.dumps({
                "error": {"code": "service_restarted", "type": "infrastructure_error"},
                "outcome": "technical_error", "label_source": "infrastructure",
            })))
        self.db.commit()

    def close(self):
        self.db.close()

    def existing(self, owner, key):
        row = self.db.execute("SELECT * FROM jobs WHERE owner=? AND idempotency_key=?", (owner, key)).fetchone()
        return dict(row) if row else None

    def create(self, owner, kind, payload, key, digest):
        job_id = uuid.uuid4().hex
        self.db.execute("""INSERT INTO jobs
            (id,owner,kind,model,request,status,created,idempotency_key,request_hash)
            VALUES(?,?,?,?,?,'queued',?,?,?)""", (
                job_id, owner, kind, payload["model"], json.dumps(payload), time.time(), key, digest))
        self.db.commit()
        return job_id

    def get(self, job_id):
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        record = dict(row)
        record["request"] = json.loads(record["request"])
        record["result"] = json.loads(record["result"]) if record["result"] else None
        return record

    def active_count(self, owner=None):
        query = "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')"
        params = ()
        if owner is not None:
            query += " AND owner=?"
            params = (owner,)
        return self.db.execute(query, params).fetchone()[0]

    def recent(self, owner, limit=30):
        rows = self.db.execute("SELECT id FROM jobs WHERE owner=? ORDER BY created DESC LIMIT ?",
                               (owner, limit)).fetchall()
        return [self.get(row["id"]) for row in rows]

    def record_count(self):
        return self.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    def start(self, job_id):
        self.db.execute("UPDATE jobs SET status='running',started=? WHERE id=?", (time.time(), job_id))
        self.db.commit()

    def finish(self, job_id, status, result):
        self.db.execute("UPDATE jobs SET status=?,finished=?,result=? WHERE id=?", (
            status, time.time(), json.dumps(result, allow_nan=False), job_id))
        self.db.commit()
