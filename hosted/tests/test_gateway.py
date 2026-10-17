import json
from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "recorder/tests"))
from test_remote import FakeAPI, CODE
from avoidance.remote import RemoteAdapter
from hosted.app import Gateway, create_app
from hosted.build_map import build

HEADERS = {"Origin": "http://testserver"}


@pytest.fixture
def setup(tmp_path):
    api = FakeAPI()
    gateway = Gateway(tmp_path, tmp_path / "missing", lambda c: RemoteAdapter(c, {"api_url": api.url, "access_code": CODE}), map_enabled=False)
    app = create_app(gateway, origin="http://testserver", secure=False)
    with TestClient(app) as client:
        gateway.health = {"online": True, "message": "Test"}
        yield api, gateway, app, client
    api.server.shutdown()


def start(client):
    response = client.post("/api/start", headers=HEADERS, json={"consent": True})
    assert response.status_code == 200, response.text
    return response.json()["conversation"]


def send(client, sid, key="request-001"):
    return client.post("/api/send", headers=HEADERS, json={"conversation": sid, "prompt": "Hello", "request_key": key})


def finish(client, sid):
    for _ in range(150):
        state = client.get("/api/state", params={"conversation": sid}).json()
        if state["job"]["status"] not in ("queued", "running"):
            return state
        time.sleep(.1)
    pytest.fail("Queue did not finish")


def test_record_and_publish_with_isolation(setup):
    api, gateway, app, first = setup
    second = TestClient(app)
    sid = start(first)
    other = start(second)
    assert second.get("/api/state", params={"conversation": sid}).status_code == 404
    assert send(second, sid).status_code == 404
    assert second.post("/api/stop", headers=HEADERS, json={"conversation": sid}).status_code == 404
    one = send(first, sid)
    assert one.status_code == 200, one.text
    assert send(first, sid).json() == one.json()
    assert send(first, sid, "request-002").status_code == 409
    assert send(second, other).status_code == 200
    for client, conversation in ((first, sid), (second, other)):
        result = finish(client, conversation)
        assert result["job"]["status"] == "complete", result
        assert len(result["rows"]) == 1
        assert result["rows"][0]["response"] == "Hello there."
        assert "path" not in json.dumps(result)
    assert len(api.requests) == 2
    assert all('under 50 words' in request['messages'][0]['content'] for request in api.requests)
    build(gateway.directory)
    payload = first.get("/api/map").json()
    assert len(payload["points"]) == 2
    assert all(point['completed_utc'] for point in payload['points'])
    assert all("label_reason" not in point for point in payload["points"])
    assert "pca_preview" in json.dumps(payload)
    assert first.get("/api/state").json()["conversation"] == sid


def test_boundary_and_offline(setup):
    _, gateway, _, client = setup
    assert client.post("/api/start", json={"consent": True}).status_code == 403
    assert client.post("/api/start", headers=HEADERS, json={"consent": False}).status_code == 400
    assert client.post("/api/start", headers={**HEADERS, "Content-Type": "application/json"}, content="{").status_code == 400
    assert client.post("/api/start", headers=HEADERS, json={"x": "a"*21000}).status_code == 413
    for path in ("/api/shutdown", "/recorder/secrets/gpu-connection.json", "/map/../recorder/secrets/gpu-connection.json", "/map/data/recordings.json"):
        assert client.get(path).status_code == 404
    assert client.post("/internal/drain", headers=HEADERS, json={}).status_code == 404
    sid = start(client)
    gateway.health = {"online": False}
    assert send(client, sid).status_code == 503
    assert client.get("/").status_code == 200
    assert client.get("/map/").status_code == 200


def test_cancel_queued_request_only(setup):
    api, gateway, app, client = setup
    api.mode = "hang"
    sid = start(client)
    send(client, sid)
    for _ in range(50):
        if gateway.service.busy():
            break
        time.sleep(.1)
    other = TestClient(app)
    second = start(other)
    send(other, second)
    assert other.post("/api/stop", headers=HEADERS, json={"conversation": second}).status_code == 200
    assert other.get("/api/state").json()["job"]["status"] == "stopped"
    assert gateway.service.busy()
    client.post("/api/stop", headers=HEADERS, json={"conversation": sid})
    assert finish(client, sid)["job"]["status"] == "stopped"


def test_restart_never_replays_pending_job(tmp_path):
    gateway = Gateway(tmp_path, tmp_path / "missing", map_enabled=False)
    gateway.health = {"online": True}
    sid, token = gateway.create_conversation(None, "test")
    gateway.submit(gateway.token_owner(token), sid, "Hello", "request-001", "test")
    gateway.db.close()
    restarted = Gateway(tmp_path, tmp_path / "missing", map_enabled=False)
    assert restarted.state(restarted.token_owner(token), sid)["job"]["status"] == "interrupted"
    restarted.db.close()


def test_model_config_can_select_a_thinking_model(tmp_path, monkeypatch):
    monkeypatch.setenv("SOFT_REFUSAL_MODEL_CONFIG", "hermes4-14b-remote.json")
    gateway = Gateway(tmp_path, tmp_path / "missing", lambda c: None, map_enabled=False)
    assert gateway.service.config["remote_model"] == "hermes"
    assert "<think>" in gateway.instruction
    assert "text-message conversation" in gateway.instruction
    gateway.db.close()


def test_resumed_conversation_uses_current_backend_instruction(setup):
    api, gateway, _, client = setup
    sid = start(client)
    gateway.service.set_instruction(sid, 'An earlier instruction.')
    assert send(client, sid).status_code == 200
    assert finish(client, sid)['job']['status'] == 'complete'
    assert api.requests[-1]['messages'][0]['content'] == gateway.instruction
