import asyncio
import hashlib
import json
import time

import pytest
from fastapi.testclient import TestClient

from soft_refusal.app import create_app
from soft_refusal.config import Settings
from soft_refusal.store import Store


TOKENS = {name: (name + "-") * 16 for name in ("alice", "bob", "operator")}


def headers(name="alice", key=None):
    value = {"Authorization": f"Bearer {TOKENS[name]}"}
    if key:
        value["Idempotency-Key"] = key
    return value


class FakeBackend:
    """Synthetic transport fixture; exercises no downloaded model or GPU."""
    loaded_model = None
    def __init__(self, delay=0.01, root=None):
        self.delay = delay
        self.root = root
        self.running = 0
        self.peak_concurrency = 0
        self.calls = []
        self.payloads = []
        self.unloads = 0

    async def run(self, job_id, payload):
        self.running += 1
        self.peak_concurrency = max(self.running, self.peak_concurrency)
        self.loaded_model = payload["model"]
        self.calls.append(payload["model"])
        self.payloads.append(payload)
        try:
            if payload.get("messages", [{}])[-1].get("content") == "hang":
                await asyncio.Event().wait()
            await asyncio.sleep(self.delay)
            if payload.get("messages", [{}])[-1].get("content") == "error":
                return {"error": {"code": "gpu_out_of_memory", "type": "OutOfMemoryError"},
                        "outcome": "technical_error", "label_source": "infrastructure",
                        "metrics": {"synthetic": True}}
            if payload["model"] == "flux":
                return {"image_available": False, "outcome": "pipeline_blocked",
                        "label_source": "pipeline_filter", "filter_signal": [True],
                        "metrics": {"synthetic": True}}
            if payload.get("record_layers") and self.root:
                import numpy as np
                path = self.root / "activations" / f"{job_id}.npz"
                with path.open("wb") as stream:
                    np.savez_compressed(stream, **{f"layer_{n}": np.full(8, n, np.float32)
                                                   for n in payload["record_layers"]})
                return {"text": "Recorded.", "outcome": "unlabeled", "label_source": None,
                        "activations": {"available": True, "layers": payload["record_layers"]},
                        "metrics": {"synthetic": True}}
            return {"text": "I cannot help with that request.", "outcome": "unlabeled",
                    "label_source": None, "finish_reason": "stop",
                    "usage": {"prompt_tokens": 4, "completion_tokens": 9, "total_tokens": 13},
                    "metrics": {"synthetic": True}}
        finally:
            self.running -= 1

    async def unload(self):
        self.unloads += 1
        self.loaded_model = None


def setup(tmp_path, **overrides):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps([{"subject": k, "sha256": hashlib.sha256(v.encode()).hexdigest(),
                                "admin": k == "operator"} for k, v in TOKENS.items()]))
    values = dict(data=tmp_path / "runtime", auth_file=auth, accept_requests=True,
                  hermes_enabled=True, job_timeout=2, idle_unload=10)
    values.update(overrides)
    return Settings(**values)


def body(model="gemma", content="Hello"):
    return {"model": model, "messages": [{"role": "user", "content": content}]}


def completed(client, job_id, actor="alice"):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        r = client.get(f"/v1/jobs/{job_id}", headers=headers(actor))
        assert r.status_code == 200
        if r.json()["status"] not in {"queued", "running"}:
            return r.json()
        time.sleep(0.01)
    raise AssertionError("Job did not finish")


def test_auth_and_private_job_access(tmp_path):
    with TestClient(create_app(setup(tmp_path), FakeBackend())) as c:
        assert c.get("/v1/health").status_code == 401
        assert c.post("/v1/chat/jobs", json=body()).status_code == 401
        job = c.post("/v1/chat/jobs", json=body(), headers=headers()).json()["job_id"]
        assert c.get(f"/v1/jobs/{job}", headers=headers("bob")).status_code == 404
        assert c.get(f"/v1/jobs/{job}/result", headers=headers("bob")).status_code == 404
        assert c.get(f"/v1/jobs/{job}", headers=headers("operator")).status_code == 200


def test_refusal_text_is_preserved_without_automatic_label(tmp_path):
    with TestClient(create_app(setup(tmp_path), FakeBackend())) as c:
        r = c.post("/v1/chat/completions", json=body(), headers=headers())
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "I cannot help with that request."
        assert r.json()["research"]["outcome"] == "unlabeled"
        assert r.json()["research"]["label_source"] is None
        assert "error" not in r.json()["research"]


def test_model_errors_are_not_refusals(tmp_path):
    with TestClient(create_app(setup(tmp_path), FakeBackend())) as c:
        r = c.post("/v1/chat/completions", json=body(content="error"), headers=headers())
        assert r.status_code == 503
        assert r.json()["status"] == "failed"
        assert r.json()["result"]["outcome"] == "technical_error"
        assert r.json()["result"]["error"]["code"] == "gpu_out_of_memory"


def test_shared_queue_serializes_chat_images_and_switching(tmp_path):
    backend = FakeBackend(delay=0.1)
    with TestClient(create_app(setup(tmp_path), backend)) as c:
        a = c.post("/v1/chat/jobs", json=body(), headers=headers()).json()["job_id"]
        b = c.post("/v1/images/jobs", json={"prompt": "An abstract paper sculpture"},
                   headers=headers("bob")).json()["job_id"]
        completed(c, a)
        image = completed(c, b, "bob")
        assert image["result"]["outcome"] == "pipeline_blocked"
        assert image["result"]["label_source"] == "pipeline_filter"
        h = c.post("/v1/chat/jobs", json=body(model="hermes"), headers=headers()).json()["job_id"]
        completed(c, h)
        assert backend.calls == ["gemma", "flux", "hermes"]
        assert backend.peak_concurrency == 1


def test_bounded_global_queue(tmp_path):
    with TestClient(create_app(setup(tmp_path, queue_limit=1), FakeBackend(delay=1))) as c:
        assert c.post("/v1/chat/jobs", json=body(), headers=headers()).status_code == 202
        assert c.post("/v1/chat/jobs", json=body(), headers=headers()).status_code == 429
        r = c.post("/v1/images/jobs", json={"prompt": "A sphere"}, headers=headers("bob"))
        assert r.status_code == 429
        assert r.json()["detail"]["code"] == "queue_full"


def test_three_jobs_same_credential_are_queued_with_ordered_chat_history(tmp_path):
    backend = FakeBackend(delay=0.15)
    with TestClient(create_app(setup(tmp_path), backend)) as c:
        a = c.post("/v1/chat/jobs", json=body(content="First question"), headers=headers()).json()["job_id"]
        image = c.post("/v1/images/jobs", json={"prompt": "A paper sculpture"}, headers=headers()).json()["job_id"]
        b = c.post("/v1/chat/jobs", json={**body(content="Follow-up"), "after_job_id": a}, headers=headers()).json()["job_id"]
        rejected = c.post("/v1/chat/jobs", json=body(), headers=headers())
        assert rejected.status_code == 429 and rejected.json()["detail"]["code"] == "per_credential_queue_full"
        completed(c, a); completed(c, image); result = completed(c, b)
        assert backend.calls == ["gemma", "flux", "gemma"] and backend.peak_concurrency == 1
        assert result["result"]["generation_settings"]["messages"] == [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "I cannot help with that request."},
            {"role": "user", "content": "Follow-up"}]


def test_dependencies_do_not_expose_foreign_jobs_or_infer_refusals(tmp_path):
    backend = FakeBackend(delay=0.05)
    with TestClient(create_app(setup(tmp_path), backend)) as c:
        a = c.post("/v1/chat/jobs", json=body(content="error"), headers=headers()).json()["job_id"]
        foreign = c.post("/v1/chat/jobs", json={**body(), "after_job_id": a}, headers=headers("bob"))
        assert foreign.status_code == 404
        b = c.post("/v1/chat/jobs", json={**body(), "after_job_id": a}, headers=headers()).json()["job_id"]
        failed = completed(c, b)
        assert failed["result"]["error"]["code"] == "chat_dependency_failed"
        assert failed["result"]["outcome"] == "technical_error" and backend.calls == ["gemma"]


def test_live_progress_and_recent_jobs_require_ownership(tmp_path):
    settings = setup(tmp_path)
    with TestClient(create_app(settings, FakeBackend(delay=0.4))) as c:
        a = c.post("/v1/chat/jobs", json=body(), headers=headers()).json()["job_id"]
        deadline = time.monotonic() + 1
        while c.get(f"/v1/jobs/{a}", headers=headers()).json()["status"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.005)
        root = settings.data / "progress"; root.mkdir()
        (root / f"{a}.json").write_text(json.dumps({"job_id": a, "phase": "generating", "text": "A partial reply"}))
        assert c.get(f"/v1/jobs/{a}", headers=headers()).json()["progress"]["text"] == "A partial reply"
        assert c.get(f"/v1/jobs/{a}", headers=headers("bob")).status_code == 404
        assert c.get("/v1/jobs", headers=headers("bob")).json()["jobs"] == []
        assert c.get("/v1/jobs").status_code == 401
        completed(c, a)
        assert c.get("/v1/jobs", headers=headers()).json()["jobs"][0]["job_id"] == a


def test_retries_are_idempotent_and_conflicts_rejected(tmp_path):
    backend = FakeBackend(delay=0.1)
    with TestClient(create_app(setup(tmp_path), backend)) as c:
        first = c.post("/v1/chat/jobs", json=body(), headers=headers(key="retry-123")).json()
        second = c.post("/v1/chat/jobs", json=body(), headers=headers(key="retry-123")).json()
        assert first["job_id"] == second["job_id"]
        assert c.post("/v1/chat/jobs", json=body(content="Different"), headers=headers(key="retry-123")).status_code == 409
        completed(c, first["job_id"])
        assert len(backend.calls) == 1


def test_timeout_releases_worker_before_next_job(tmp_path):
    backend = FakeBackend()
    with TestClient(create_app(setup(tmp_path, job_timeout=0.1), backend)) as c:
        job = c.post("/v1/chat/jobs", json=body(content="hang"), headers=headers()).json()["job_id"]
        result = completed(c, job)
        assert result["result"]["error"]["code"] == "model_timeout"
        assert backend.unloads >= 1 and backend.running == 0
        r = c.post("/v1/chat/completions", json=body(), headers=headers("bob"))
        assert r.status_code == 200
        assert backend.peak_concurrency == 1


def test_restart_preserves_completed_and_marks_unfinished(tmp_path):
    settings = setup(tmp_path)
    with TestClient(create_app(settings, FakeBackend())) as c:
        done = c.post("/v1/chat/jobs", json=body(), headers=headers()).json()["job_id"]
        completed(c, done)
    store = Store(settings.data)
    unfinished = store.create("bob", "chat", body(), None, "unused")
    store.close()
    with TestClient(create_app(settings, FakeBackend())) as c:
        assert c.get(f"/v1/jobs/{done}", headers=headers()).json()["status"] == "succeeded"
        r = c.get(f"/v1/jobs/{unfinished}", headers=headers("bob")).json()
        assert r["status"] == "interrupted"
        assert r["result"]["error"]["code"] == "service_restarted"


def test_rejects_oversized_invalid_and_streaming_requests(tmp_path):
    with TestClient(create_app(setup(tmp_path), FakeBackend())) as c:
        assert c.post("/v1/chat/jobs", json=body(content="x" * 12001), headers=headers()).status_code == 422
        r = c.post("/v1/chat/jobs", json={**body(), "stream": True}, headers=headers())
        assert r.status_code == 422
        assert c.post("/v1/chat/jobs", content=b"x" * 65537, headers=headers()).status_code == 413
        assert c.post("/v1/images/jobs", json={"prompt": "Hello", "width": 1000}, headers=headers()).status_code == 422
        assert c.post("/v1/chat/jobs", json={"model": "unknown", "messages": body()["messages"]}, headers=headers()).status_code == 422


def test_paused_default_and_admin_pause(tmp_path):
    with TestClient(create_app(setup(tmp_path, accept_requests=False), FakeBackend())) as c:
        assert c.post("/v1/chat/jobs", json=body(), headers=headers()).status_code == 503
        assert c.get("/healthz").json() == {"status": "alive"}
        assert c.get("/v1/health", headers=headers()).json()["status"] == "paused"
    with TestClient(create_app(setup(tmp_path), FakeBackend())) as c:
        assert c.post("/v1/admin/pause", headers=headers()).status_code == 403
        assert c.post("/v1/admin/pause", headers=headers("operator")).status_code == 200
        assert c.post("/v1/chat/jobs", json=body(), headers=headers()).status_code == 503


def test_optional_hermes_is_explicitly_enabled(tmp_path):
    with TestClient(create_app(setup(tmp_path, hermes_enabled=False), FakeBackend())) as c:
        r = c.post("/v1/chat/jobs", json=body(model="hermes"), headers=headers())
        assert r.status_code == 409
        assert r.json()["detail"]["code"] == "comparison_model_not_enabled"


def test_no_duplicate_website_is_served(tmp_path):
    with TestClient(create_app(setup(tmp_path), FakeBackend())) as c:
        assert c.get("/").status_code == 404
        assert c.get("/docs").status_code == 404
        assert c.get("/v1/openapi.json").status_code == 401


def test_recorded_activations_are_private_and_downloadable(tmp_path):
    import io
    import numpy as np
    settings = setup(tmp_path)
    with TestClient(create_app(settings, FakeBackend(root=settings.data))) as c:
        plain = c.post("/v1/chat/jobs", json=body(), headers=headers()).json()["job_id"]
        completed(c, plain)
        assert c.get(f"/v1/jobs/{plain}/activations", headers=headers()).json()["detail"]["code"] == "activations_not_recorded"
        job = c.post("/v1/chat/jobs", json={**body(), "record_layers": [3, 1], "trace": True},
                     headers=headers()).json()["job_id"]
        completed(c, job)
        assert c.get(f"/v1/jobs/{job}/activations", headers=headers("bob")).status_code == 404
        r = c.get(f"/v1/jobs/{job}/activations", headers=headers())
        assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
        with np.load(io.BytesIO(r.content), allow_pickle=False) as data:
            assert sorted(data.files) == ["layer_1", "layer_3"] and data["layer_3"][0] == 3


def test_record_layers_are_validated(tmp_path):
    with TestClient(create_app(setup(tmp_path), FakeBackend())) as c:
        for layers in ([], [1, 1], [-1], [128], list(range(9))):
            r = c.post("/v1/chat/jobs", json={**body(), "record_layers": layers}, headers=headers())
            assert r.status_code == 422, layers


def test_cancel_queued_and_running_jobs(tmp_path):
    settings = setup(tmp_path)
    backend = FakeBackend(delay=0.4)
    with TestClient(create_app(settings, backend)) as c:
        running = c.post("/v1/chat/jobs", json=body(), headers=headers()).json()["job_id"]
        queued = c.post("/v1/chat/jobs", json=body(content="Second"), headers=headers()).json()["job_id"]
        deadline = time.monotonic() + 1
        while c.get(f"/v1/jobs/{running}", headers=headers()).json()["status"] != "running":
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert c.post(f"/v1/jobs/{queued}/cancel", headers=headers("bob")).status_code == 404
        r = c.post(f"/v1/jobs/{queued}/cancel", headers=headers()).json()
        assert r["status"] == "cancelled" and r["result"]["error"]["code"] == "cancelled_by_user"
        assert r["result"]["outcome"] == "cancelled"
        c.post(f"/v1/jobs/{running}/cancel", headers=headers())
        # The separate model process polls this marker between tokens.
        assert (settings.data / "cancel" / running).exists()
        completed(c, running)
        assert not (settings.data / "cancel" / running).exists()
        assert backend.calls == ["gemma"]


def test_empty_earlier_reply_is_kept_but_empty_user_text_rejected(tmp_path):
    with TestClient(create_app(setup(tmp_path), FakeBackend())) as c:
        history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""},
                   {"role": "user", "content": "Again"}]
        assert c.post("/v1/chat/jobs", json={"model": "gemma", "messages": history}, headers=headers()).status_code == 202
        assert c.post("/v1/chat/jobs", json=body(content=""), headers=headers()).status_code == 422
