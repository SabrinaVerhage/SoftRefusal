import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from collections import deque

from fastapi import HTTPException
from .progress import clear_cancel, request_cancel
from .schemas import ChatRequest

# Request problems and user cancellations leave a healthy resident model loaded.
KEEP_WORKER = {"input_too_long", "invalid_chat_template", "invalid_record_layers", "cancelled_by_user"}


def failure(code, error_type="infrastructure_error", **extra):
    return {"outcome": "technical_error", "label_source": "infrastructure",
            "error": {"code": code, "type": error_type}, **extra}


class ProcessBackend:
    """One resident model process. Switching/timeout destroys its CUDA context."""

    def __init__(self, settings):
        self.settings = settings
        self.process = None
        self.loaded_model = None

    async def unload(self):
        if self.process:
            if self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), 10)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            self.process = None
            self.loaded_model = None

    async def run(self, job_id, payload):
        if not self.process or self.process.returncode is not None or self.loaded_model != payload["model"]:
            await self.unload()
            env = os.environ.copy()
            env["SOFT_REFUSAL_DATA"] = str(self.settings.data.resolve())
            env["SOFT_REFUSAL_CHAT_INPUT_LIMIT"] = str(self.settings.chat_input_limit)
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-m", "soft_refusal.worker", "--model", payload["model"],
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                # No model output/prompt/credentials in the public server logs.
                stderr=asyncio.subprocess.DEVNULL, env=env, limit=16 * 1024 * 1024)
            self.loaded_model = payload["model"]
        self.process.stdin.write((json.dumps({"id": job_id, "payload": payload}) + "\n").encode())
        await self.process.stdin.drain()
        line = await self.process.stdout.readline()
        if not line:
            raise RuntimeError("model_worker_exited")
        result = json.loads(line)
        if result.get("job_id") != job_id:
            raise RuntimeError("model_worker_protocol_error")
        return result


class GPUQueue:
    def __init__(self, settings, store, backend):
        self.settings, self.store, self.backend = settings, store, backend
        self.pending = deque()
        self.changed = asyncio.Event()
        self.events = {}
        self.accepting = settings.accept_requests
        self.closed = False
        self.active = None
        self.task = None
        self.last_metrics = None
        self.rate = {}

    def start(self):
        self.task = asyncio.create_task(self._work())

    def submit(self, principal, kind, payload, key=None):
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if key:
            if len(key) > 128 or any(ord(c) < 32 or ord(c) > 126 for c in key):
                raise HTTPException(422, detail={"code": "invalid_idempotency_key"})
            previous = self.store.existing(principal.subject, key)
            if previous:
                if previous["request_hash"] != digest:
                    raise HTTPException(409, detail={"code": "idempotency_conflict"})
                return previous["id"]
        if not self.accepting:
            raise HTTPException(503, detail={"code": "submissions_paused"})
        if payload["model"] == "hermes" and not self.settings.hermes_enabled:
            raise HTTPException(409, detail={"code": "comparison_model_not_enabled"})
        if self.store.record_count() >= 10000 or shutil.disk_usage(self.settings.data).free < 2 * 1024**3:
            raise HTTPException(503, detail={"code": "storage_capacity_reached"})
        parent_id = payload.get("after_job_id")
        if parent_id:
            parent = self.store.get(parent_id)
            if not parent or parent["owner"] != principal.subject:
                raise HTTPException(404, detail={"code": "job_not_found"})
            if parent["kind"] != "chat" or parent["model"] != payload["model"]:
                raise HTTPException(422, detail={"code": "invalid_chat_dependency"})
        if self.store.active_count(principal.subject) >= self.settings.per_credential_limit:
            raise HTTPException(429, detail={"code": "per_credential_queue_full",
                                           "limit": self.settings.per_credential_limit}, headers={"Retry-After": "5"})
        if self.store.active_count() >= self.settings.queue_limit:
            raise HTTPException(429, detail={"code": "queue_full"}, headers={"Retry-After": "5"})
        now = time.monotonic()
        timestamps = self.rate.setdefault(principal.subject, deque())
        while timestamps and timestamps[0] <= now - 60:
            timestamps.popleft()
        if len(timestamps) >= 6:
            raise HTTPException(429, detail={"code": "submission_rate_limit"}, headers={"Retry-After": "60"})
        job_id = self.store.create(principal.subject, kind, payload, key, digest)
        timestamps.append(now)
        self.events[job_id] = asyncio.Event()
        self.pending.append(job_id)
        self.changed.set()
        return job_id

    def position(self, job_id):
        if self.active == job_id:
            return 0
        try:
            return list(self.pending).index(job_id) + 1
        except ValueError:
            return None

    async def wait(self, job_id):
        event = self.events.get(job_id)
        if event:
            # Shield work from HTTP client disconnects; it stays in the same queue.
            await event.wait()
        return self.store.get(job_id)

    def cancel(self, job_id):
        """Queued jobs leave the queue now; a running job stops at its next token."""
        if job_id in self.pending:
            self.pending.remove(job_id)
            self.complete(job_id, "cancelled", {"outcome": "cancelled", "label_source": "user",
                                                "error": {"code": "cancelled_by_user", "type": "user_request"}})
        elif self.active == job_id:
            request_cancel(self.settings.data, job_id)

    def complete(self, job_id, status, result):
        clear_cancel(self.settings.data, job_id)
        self.store.finish(job_id, status, result)
        self.last_metrics = result.get("metrics")
        event = self.events.pop(job_id, None)
        if event:
            event.set()

    async def _work(self):
        try:
            while not self.closed:
                if not self.pending:
                    self.changed.clear()
                    try:
                        await asyncio.wait_for(self.changed.wait(), self.settings.idle_unload)
                    except asyncio.TimeoutError:
                        await self.backend.unload()
                    continue
                job_id = self.pending.popleft()
                record = self.store.get(job_id)
                waited = time.time() - record["created"]
                if waited > self.settings.queue_timeout:
                    self.complete(job_id, "failed", failure("queue_expired", queue_wait_seconds=waited))
                    continue
                self.active = job_id
                self.store.start(job_id)
                started = time.monotonic()
                try:
                    payload = record["request"]
                    parent_id = payload.get("after_job_id")
                    if parent_id:
                        parent = self.store.get(parent_id)
                        if not parent or parent["status"] != "succeeded" or not isinstance(parent["result"].get("text"), str):
                            self.complete(job_id, "failed", failure("chat_dependency_failed", parent_job_id=parent_id))
                            continue
                        previous = parent["result"].get("generation_settings", parent["request"])
                        payload = {**payload, "after_job_id": None, "messages": [*previous["messages"],
                                   {"role": "assistant", "content": parent["result"]["text"]}, *payload["messages"]]}
                        try:
                            ChatRequest.model_validate(payload)
                        except ValueError:
                            self.complete(job_id, "failed", failure("conversation_limit_exceeded",
                                          max_messages=64, max_characters=24000, parent_job_id=parent_id))
                            continue
                    result = await asyncio.wait_for(
                        self.backend.run(job_id, payload), self.settings.job_timeout)
                    result.setdefault("generation_settings", payload)
                    result.setdefault("metrics", {}).update({
                        "queue_wait_seconds": waited,
                        "worker_roundtrip_seconds": time.monotonic() - started,
                    })
                    code = (result.get("error") or {}).get("code")
                    status = "cancelled" if code == "cancelled_by_user" else "failed" if code else "succeeded"
                    self.complete(job_id, status, result)
                    if code and code not in KEEP_WORKER:
                        await self.backend.unload()
                except asyncio.TimeoutError:
                    await self.backend.unload()
                    self.complete(job_id, "failed", failure("model_timeout", queue_wait_seconds=waited))
                except asyncio.CancelledError:
                    await self.backend.unload()
                    self.complete(job_id, "interrupted", failure("service_stopped"))
                    raise
                except Exception:
                    await self.backend.unload()
                    self.complete(job_id, "failed", failure("model_worker_failed"))
                finally:
                    self.active = None
        finally:
            await self.backend.unload()

    async def close(self):
        self.accepting = False
        self.closed = True
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        for job_id in self.pending:
            self.complete(job_id, "interrupted", failure("service_stopped"))
        self.pending.clear()

