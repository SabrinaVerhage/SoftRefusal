import json
import time
import urllib.error

from fastapi.testclient import TestClient

from hosted.app import Gateway, create_app

HEADERS = {"Origin": "http://testserver"}
PNG = b"\x89PNG\r\n\x1a\nsynthetic-test-image"


def make(tmp_path):
    g = Gateway(tmp_path, tmp_path / "missing", adapter_factory=lambda c: None, map_enabled=False)
    app = create_app(g, origin="http://testserver", secure=False)
    return g, app


def enter(client):
    assert client.post("/api/images/session", headers=HEADERS, json={"consent": True}).status_code == 200


def submit(client, key="image-test-001", **kwargs):
    return client.post("/api/images/jobs", headers=HEADERS, json={"prompt": "A blue ceramic cup", "request_key": key, **kwargs})


def finish(client):
    for _ in range(100):
        jobs = client.get("/api/images").json()["jobs"]
        if jobs and jobs[0]["status"] not in {"queued", "running"}:
            return jobs[0]
        time.sleep(.05)
    raise AssertionError("Image job did not finish")


def test_image_ownership_idempotency_and_offline_results(tmp_path):
    g, app = make(tmp_path)
    calls = []
    def request(path, body=None, key=None, binary=False):
        calls.append((path, body, key))
        if binary:
            return PNG
        return {"job_id": "a" * 32, "status": "succeeded", "result": {"image_available": True}}
    g.images.request = request
    with TestClient(app) as first:
        second = TestClient(app)
        assert submit(first).status_code == 401
        assert first.post("/api/images/session", json={"consent": True}).status_code == 403
        assert first.post("/api/images/session", headers=HEADERS, json={"consent": False}).status_code == 400
        enter(first)
        enter(second)
        g.images.online = True
        one = submit(first)
        assert one.status_code == 200
        assert submit(first).json() == one.json()
        assert submit(first, seed=43).status_code == 409
        job = finish(first)
        assert job["status"] == "complete"
        assert first.get(job["result_url"]).content == PNG
        assert second.get(job["result_url"]).status_code == 404
        assert second.get("/api/images").json()["jobs"] == []
        assert second.get("/api/gallery").json()["images"] == []
        assert second.get(f"/api/gallery/{job['id']}/image").status_code == 404
        assert len([c for c in calls if c[1] is not None]) == 1
        assert "upstream_id" not in json.dumps(job)
        g.images.online = False
        assert first.get(job["result_url"]).content == PNG
        assert submit(first, key="offline-002").status_code == 503


def test_validation_shared_queue_and_drain(tmp_path):
    g, app = make(tmp_path)
    # Do not start workers: inspect durable admission before any processing.
    g.start = lambda: None
    with TestClient(app) as client:
        enter(client)
        g.images.online = True
        g.health = {"online": True}
        assert submit(client, width=513).status_code == 400
        assert submit(client, seed=True).status_code == 400
        assert submit(client, seed=-1).status_code == 400
        sid = client.post("/api/start", headers=HEADERS, json={"consent": True}).json()["conversation"]
        assert submit(client).status_code == 200
        assert client.get("/api/images").json()["jobs"][0]["position"] == 1
        assert client.post("/api/send", headers=HEADERS, json={"conversation": sid, "prompt": "Hello", "request_key": "chat-test-001"}).status_code == 409
        assert g.db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] == 1
        g.accepting = False
        other = TestClient(app)
        assert other.post("/api/images/session", headers=HEADERS, json={"consent": True}).status_code == 503
    restarted = Gateway(tmp_path, tmp_path / "missing", map_enabled=False)
    assert restarted.db.execute("SELECT status FROM jobs").fetchone()[0] == "interrupted"
    restarted.db.close()


def test_image_failure_is_not_refusal(tmp_path):
    g, app = make(tmp_path)
    g.images.request = lambda *a, **kw: {"job_id": "b" * 32, "status": "failed", "result": {"error": {"code": "worker_error"}}}
    with TestClient(app) as client:
        enter(client)
        g.images.online = True
        submit(client)
        job = finish(client)
        assert job["status"] == "error"
        assert "worker_error" in job["error"] and "not a refusal" in job["error"]
        assert "result_url" not in job


def test_hermes_uses_isolated_recordings_and_instruction(tmp_path, monkeypatch):
    gemma, _ = make(tmp_path)
    old_sid, _ = gemma.create_conversation(None, "test")
    gemma.db.close()
    monkeypatch.setenv("SOFT_REFUSAL_MODEL_CONFIG", "hermes4-14b-remote.json")
    hermes, _ = make(tmp_path)
    assert hermes.directory == tmp_path / "models/hermes"
    assert old_sid not in hermes.service.sessions
    assert (tmp_path / "recordings" / old_sid / "session.json").exists()
    sid, token = hermes.create_conversation(None, "test")
    assert "<think>" in hermes.service.sessions[sid]["instruction"]
    assert hermes.service.config["layers"] == [10, 20, 26, 35]
    assert hermes.service.auto_label_enabled is False
    assert hermes.state(hermes.token_owner(token))["model"]["id"] == "NousResearch/Hermes-4-14B"
    hermes.db.close()


def test_shared_gallery_persists_without_gpu_or_cookie(tmp_path):
    g, app = make(tmp_path)
    g.images.request = lambda *a, binary=False, **kw: PNG if binary else {
        "job_id": "c" * 32, "status": "succeeded", "result": {"image_available": True}}
    with TestClient(app) as client:
        enter(client)
        g.images.online = True
        assert submit(client, share_gallery="true").status_code == 400
        assert submit(client, share_gallery=True).status_code == 200
        job = finish(client)
        assert job["shared"] is True
        assert submit(client, share_gallery=False).status_code == 409
        g.images.online = False
        anonymous = TestClient(app)
        gallery = anonymous.get("/api/gallery").json()
        assert len(gallery["images"]) == 1
        item = gallery["images"][0]
        assert item["prompt"] == "A blue ceramic cup"
        assert not {"owner", "upstream_id", "payload"} & item.keys()
        assert anonymous.get(item["image_url"]).content == PNG
        assert anonymous.get(job["result_url"]).status_code == 404
        assert anonymous.get("/gallery/").status_code == 200
    g.db.close()
    restarted, new_app = make(tmp_path)
    with TestClient(new_app) as visitor:
        assert visitor.get("/api/gallery").json()["images"][0]["id"] == job["id"]
        assert visitor.get(item["image_url"]).content == PNG
        (tmp_path / "images" / (job["id"] + ".png")).unlink()
        assert visitor.get(item["image_url"]).status_code == 404


def test_gallery_pagination_excludes_private_and_incomplete(tmp_path):
    g, app = make(tmp_path)
    g.start = lambda: None
    payload = json.dumps({"width": 512, "height": 512, "seed": 42})
    for i in range(28):
        g.db.execute("INSERT INTO jobs(id,owner,conversation,prompt,status,created,request_key,kind,payload,shared) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (str(i), "owner", "", "test", "running" if i == 27 else "complete", i, str(i), "image", payload, 0 if i == 26 else 1))
    g.db.commit()
    with TestClient(app) as client:
        first = client.get("/api/gallery").json()
        assert len(first["images"]) == 24
        assert first["images"][0]["id"] == "25"
        second = client.get(f"/api/gallery?offset={first['next_offset']}").json()
        assert [row["id"] for row in second["images"]] == ["1", "0"]
        assert second["next_offset"] is None
        assert client.get("/api/gallery?offset=-1").status_code == 400
        assert client.get("/api/gallery/27/image").status_code == 404
