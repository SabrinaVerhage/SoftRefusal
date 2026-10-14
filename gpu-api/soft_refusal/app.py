import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .auth import Auth, Principal
from .config import MODELS, Settings
from .queue import GPUQueue, ProcessBackend
from .schemas import ChatRequest, ImageRequest
from .store import Store
from .progress import read_progress


class SingleInstance:
    """OS lock prevents a second API process from creating a second GPU queue."""
    def __init__(self, path):
        self.file = path.open("a+b")
        if os.name == "nt":
            import msvcrt
            self.file.write(b"0")
            self.file.flush()
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def close(self):
        self.file.close()


class BoundedBody:
    def __init__(self, app, max_bytes=65536):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        chunks, length = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            length += len(message.get("body", b""))
            if length > self.max_bytes:
                response = JSONResponse({"error": {"code": "request_too_large"}}, status_code=413)
                return await response(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        async def replay():
            return chunks.pop(0) if chunks else await receive()
        await self.app(scope, replay, send)


def create_app(settings=None, backend=None):
    settings = settings or Settings()
    auth = Auth(settings.auth_file)

    @asynccontextmanager
    async def lifespan(app):
        settings.data.mkdir(parents=True, exist_ok=True)
        lock = SingleInstance(settings.data / "supervisor.lock")
        try:
            store = Store(settings.data)
            queue = GPUQueue(settings, store, backend or ProcessBackend(settings))
            app.state.store, app.state.queue = store, queue
            queue.start()
            try:
                yield
            finally:
                await queue.close()
                store.close()
        finally:
            lock.close()

    app = FastAPI(title="Soft Refusal model API", version="0.3.0",
                  lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BoundedBody)
    if settings.origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.origins,
                           allow_credentials=False, allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type", "Idempotency-Key"])

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # FastAPI defaults echo invalid input; avoid reflecting research prompts.
        return JSONResponse({"error": {"code": "invalid_request", "fields": [
            {"loc": e["loc"], "type": e["type"], "message": e["msg"]} for e in exc.errors()
        ]}}, status_code=422)

    @app.middleware("http")
    async def privacy_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        return response

    @app.get("/healthz")
    async def liveness():
        return {"status": "alive"}

    @app.get("/v1/health")
    async def health(principal: Principal = Depends(auth)):
        queue = app.state.queue
        return {"status": "accepting" if queue.accepting else "paused",
                "queue_size": len(queue.pending), "running": queue.active is not None,
                "resident_model": queue.backend.loaded_model,
                "precision": "bfloat16", "quantization": None,
                "limits": {"queue": settings.queue_limit, "per_credential": settings.per_credential_limit,
                           "chat_input_tokens": settings.chat_input_limit, "chat_messages": 64,
                           "chat_characters": 24000, "idle_unload_seconds": settings.idle_unload},
                "models": {k: {**v, "enabled": k != "hermes" or settings.hermes_enabled}
                           for k, v in MODELS.items()},
                "last_metrics": queue.last_metrics,
                "gpu_validation": "requires_recorded_live_rehearsal"}

    def owned(job_id, principal):
        record = app.state.store.get(job_id)
        if not record or (record["owner"] != principal.subject and not principal.admin):
            raise HTTPException(404, detail={"code": "job_not_found"})
        return record

    def view(record):
        return {"job_id": record["id"], "kind": record["kind"], "model": record["model"],
                "status": record["status"], "queue_position": app.state.queue.position(record["id"]),
                "created_at": record["created"], "started_at": record["started"],
                "finished_at": record["finished"], "result": record["result"],
                "progress": read_progress(settings.data, record["id"]) if record["status"] == "running" else None,
                "status_url": f"/v1/jobs/{record['id']}",
                "result_url": f"/v1/jobs/{record['id']}/result"}

    def submit(request, principal, kind, body):
        job_id = app.state.queue.submit(principal, kind, body.model_dump(), request.headers.get("idempotency-key"))
        return job_id

    @app.post("/v1/chat/jobs", status_code=202)
    async def chat_job(body: ChatRequest, request: Request, principal: Principal = Depends(auth)):
        return view(owned(submit(request, principal, "chat", body), principal))

    @app.post("/v1/images/jobs", status_code=202)
    async def image_job(body: ImageRequest, request: Request, principal: Principal = Depends(auth)):
        return view(owned(submit(request, principal, "image", body), principal))

    @app.post("/v1/chat/completions")
    async def chat_completion(body: ChatRequest, request: Request, principal: Principal = Depends(auth)):
        job_id = submit(request, principal, "chat", body)
        record = await app.state.queue.wait(job_id)
        result = record["result"]
        if record["status"] != "succeeded":
            code = result["error"]["code"]
            status = 422 if code in {"input_too_long", "invalid_chat_template"} else 503
            return JSONResponse(view(record), status_code=status)
        return {"id": job_id, "object": "chat.completion", "created": int(record["created"]),
                "model": MODELS[body.model]["repo"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]},
                             "finish_reason": result["finish_reason"]}],
                "usage": result["usage"], "research": result,
                "status_url": f"/v1/jobs/{job_id}"}

    @app.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str, principal: Principal = Depends(auth)):
        return view(owned(job_id, principal))

    @app.get("/v1/jobs")
    async def recent_jobs(limit: int = 30, principal: Principal = Depends(auth)):
        if not 1 <= limit <= 50:
            raise HTTPException(422, detail={"code": "invalid_limit"})
        return {"jobs": [view(record) for record in app.state.store.recent(principal.subject, limit)]}

    @app.get("/v1/jobs/{job_id}/result")
    async def job_result(job_id: str, principal: Principal = Depends(auth)):
        record = owned(job_id, principal)
        if record["status"] in ("queued", "running"):
            return JSONResponse(view(record), status_code=202)
        if record["status"] != "succeeded" or record["kind"] != "image":
            return record["result"]
        if record["result"].get("image_available"):
            path = settings.data / "images" / f"{record['id']}.png"
            if not path.is_file():
                raise HTTPException(410, detail={"code": "result_file_missing"})
            return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})
        return record["result"]

    @app.get("/v1/jobs/{job_id}/activations")
    async def job_activations(job_id: str, principal: Principal = Depends(auth)):
        record = owned(job_id, principal)
        if record["status"] in ("queued", "running"):
            return JSONResponse(view(record), status_code=202)
        if record["status"] != "succeeded" or not (record["result"].get("activations") or {}).get("available"):
            raise HTTPException(404, detail={"code": "activations_not_recorded"})
        path = settings.data / "activations" / f"{record['id']}.npz"
        if not path.is_file():
            raise HTTPException(410, detail={"code": "result_file_missing"})
        return FileResponse(path, media_type="application/octet-stream",
                            headers={"Cache-Control": "private, no-store"})

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, principal: Principal = Depends(auth)):
        record = owned(job_id, principal)
        if record["status"] in ("queued", "running"):
            app.state.queue.cancel(job_id)
        return view(app.state.store.get(job_id))

    @app.get("/v1/openapi.json")
    async def schema(principal: Principal = Depends(auth)):
        return app.openapi()

    @app.post("/v1/admin/pause")
    async def pause(principal: Principal = Depends(auth)):
        if not principal.admin:
            raise HTTPException(403, detail={"code": "admin_required"})
        app.state.queue.accepting = False
        return {"accepting": False, "active_jobs": app.state.store.active_count(),
                "compute_billing": "unchanged; stop the Pod in RunPod to end compute charges"}

    if settings.web_root:
        root = Path(settings.web_root).resolve()
        if not (root / "index.html").is_file():
            raise ValueError("WEB_ROOT must be an inspected public-assets-only directory with index.html")
        app.mount("/", StaticFiles(directory=root, html=True), name="existing-website")
    return app


def main():
    import uvicorn
    # Loopback by default. Expose 8000 through RunPod HTTPS only after auth review.
    uvicorn.run(create_app(), host=os.getenv("SOFT_REFUSAL_BIND", "127.0.0.1"),
                port=int(os.getenv("SOFT_REFUSAL_PORT", "8000")), workers=1,
                access_log=False, limit_concurrency=64, timeout_keep_alive=5)


if __name__ == "__main__":
    main()
