import io
import json
import threading
import time
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer

import pytest

from avoidance.chat import ChatService, make_handler
from avoidance.io import load_dataset, read_json
from avoidance.model import GenerationCancelled
from test_pipeline import tiny_adapter


def wait_for_job(service):
    deadline = time.monotonic() + 30
    while service.job["status"] == "running":
        if time.monotonic() > deadline:
            pytest.fail("Background job did not finish")
        time.sleep(.02)
    return service.job


def settings(adapter):
    return dict(adapter.config, model_id="TINY-TEST-FIXTURE", revision="a" * 40, adapter="gemma3", dtype="float32", device="cpu", seed=42)


def test_record_resume_export_and_real_event_order(tmp_path, tiny_adapter):
    service = ChatService(settings(tiny_adapter), tmp_path, lambda _: tiny_adapter)
    sid = service.create()
    service.submit(sid, "hello world")
    assert wait_for_job(service)["status"] == "complete"
    events = [e["kind"] for e in service.job["events"]]
    assert events.index("capturing") < events.index("snapshot") < events.index("generating") < events.index("saving") < events.index("saved")
    assert "progress" in events and "text" in events
    first = service.sessions[sid]["latest"]
    _, rows, arrays = load_dataset(tmp_path / sid / first)
    assert arrays["layer_1"].shape == (1, 16)
    assert rows[0]["snapshot_preview"][0]["dimensions"] == 16
    service.submit(sid, "hello")
    assert wait_for_job(service)["status"] == "complete"
    latest = service.sessions[sid]["latest"]
    _, rows, arrays = load_dataset(tmp_path / sid / latest)
    assert len(rows) == 2 and arrays["layer_1"].shape == (2, 16)
    assert len(rows[1]["messages"]) == 3
    assert len(load_dataset(tmp_path / sid / first)[1]) == 1
    restarted = ChatService(settings(tiny_adapter), tmp_path, lambda _: tiny_adapter)
    assert len(restarted.state(sid)["rows"]) == 2
    with zipfile.ZipFile(io.BytesIO(restarted.export(sid))) as z:
        assert {"manifest.json", "activations.npz", "steps.jsonl", "annotations.csv"} <= set(z.namelist())
        assert any(name.startswith("recording-log/") for name in z.namelist())


def test_stopped_turn_and_concurrent_request(tmp_path, tiny_adapter):
    ready = threading.Event()
    class Slow:
        def step(self, messages, on_event, cancel_event, **kwargs):
            ready.set()
            on_event("generating", message="Test fixture waiting")
            cancel_event.wait(10)
            raise GenerationCancelled("Stopped for test")
    service = ChatService(settings(tiny_adapter), tmp_path, lambda _: Slow())
    sid = service.create()
    service.submit(sid, "hello")
    assert ready.wait(10)
    with pytest.raises(ValueError, match="already working"):
        service.submit(sid, "another")
    service.stop()
    assert wait_for_job(service)["status"] == "stopped"
    assert service.sessions[sid]["latest"] is None
    assert service.state(sid)["rows"] == []
    assert list((tmp_path / sid / "attempts").glob("*/result.json"))


def test_failure_preserves_previous_checkpoint(tmp_path, tiny_adapter, monkeypatch):
    service = ChatService(settings(tiny_adapter), tmp_path, lambda _: tiny_adapter)
    sid = service.create()
    service.submit(sid, "hello")
    assert wait_for_job(service)["status"] == "complete"
    first = service.sessions[sid]["latest"]
    def fail(*args, **kwargs):
        raise OSError("simulated recording failure")
    monkeypatch.setattr(service, "_save", fail)
    service.submit(sid, "world")
    assert wait_for_job(service)["status"] == "error"
    assert service.sessions[sid]["latest"] == first
    assert len(load_dataset(tmp_path / sid / first)[1]) == 1


def test_http_origin_guard_and_assets(tmp_path, tiny_adapter):
    service = ChatService(settings(tiny_adapter), tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(url) as r:
            assert b"The recording desk" in r.read()
            assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
        request = urllib.request.Request(url + "/api/new", data=b"{}", headers={"Content-Type": "application/json", "Origin": "https://unrelated.example"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(request)
        assert e.value.code == 403
        request = urllib.request.Request(url + "/api/new", data=b"{}", headers={"Content-Type": "application/json", "Origin": url})
        with urllib.request.urlopen(request) as r:
            assert json.load(r)["session"] in service.sessions
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(url + "/../../README.md")
        assert e.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
