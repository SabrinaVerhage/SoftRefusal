"""Recording room ↔ cloud API, against a local stand-in for the soft-refusal backend.
No GPU, model or network access: the fake server speaks the backend's HTTP contract."""
import hashlib
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from avoidance.chat import ChatService
from avoidance.io import load_dataset, read_json
from avoidance.remote import RemoteAdapter, load_connection

CODE = "test-access-code"
CONFIG = read_json("configs/gemma3-12b-remote.json")
DIMS = 8


def vectors_for(layers):
    return {layer: np.arange(DIMS, dtype=np.float32) + layer for layer in layers}


class FakeAPI:
    def __init__(self, mode="ok", config=CONFIG):
        self.mode, self.jobs, self.requests, self.cancelled = mode, {}, [], []
        self.config, self.chat_output = config, "Hello there."
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, payload, raw=False):
                body = payload if raw else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/octet-stream" if raw else "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def authorized(self):
                if self.headers.get("Authorization") != f"Bearer {CODE}":
                    self.send(401, {"detail": {"code": "unauthorized"}})
                    return False
                return True

            def do_GET(self):
                if not self.authorized():
                    return
                if self.path == "/v1/health":
                    return self.send(200, {"status": "accepting", "models": {api.config["remote_model"]: {"enabled": True}}})
                parts = self.path.strip("/").split("/")
                job = api.jobs[parts[2]]
                if parts[-1] == "activations":
                    return self.send(200, job["npz"], raw=True)
                return self.send(200, api.advance(job))

            def do_POST(self):
                if not self.authorized():
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else {}
                if self.path == "/v1/chat/jobs":
                    api.requests.append(body)
                    job = {"job_id": f"{len(api.jobs):032x}", "status": "queued", "polls": 0, "body": body,
                           "queue_position": 1, "result": None, "progress": None}
                    api.jobs[job["job_id"]] = job
                    return self.send(202, api.view(job))
                job_id = self.path.strip("/").split("/")[2]
                api.cancelled.append(job_id)
                api.jobs[job_id]["status"] = "cancelled"
                return self.send(200, api.view(api.jobs[job_id]))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def view(self, job):
        return {k: job[k] for k in ("job_id", "status", "queue_position", "result", "progress")}

    def advance(self, job):
        job["polls"] += 1
        if "record_layers" not in job["body"]:
            job.update(status="succeeded", result={"model": self.config["model_id"], "revision": self.config["revision"],
                       "text": getattr(self, "judge_output", '{"outcome":"answered","reason":"The requested greeting was provided."}'),
                       "hit_token_limit": False})
            return self.view(job)
        layers = job["body"]["record_layers"]
        previews = [{"layer": n, "dimensions": DIMS, "preview": [float(n)] * 8} for n in layers]
        candidates = [{"step": i, "top": [{"id": 4 + i, "piece": f"t{i}", "probability": .9}],
                       "selection": "greedy_argmax_after_generation_processors"} for i in range(3)]
        trace = {"tokens": {"total": 3, "shown": [], "scope": "last_32_formatted_prompt_tokens"},
                 "layers": [{"layer": i, "magnitude": 1.0, "preview": [0.0] * 16, "total_layers": 48} for i in range(48)]}
        if job["status"] == "cancelled":
            return self.view(job)
        if self.mode == "hang" or job["polls"] == 1:
            job["status"] = "running"
            job["progress"] = {"phase": "generating", "input_tokens": 3, "text": "Hel", "output_tokens": 2,
                               "trace": {**trace, "snapshots": previews, "candidates": candidates[:2]}}
            return self.view(job)
        if self.mode == "too_long":
            job.update(status="failed", progress=None, result={"outcome": "technical_error", "error": {
                "code": "input_too_long", "input_tokens": 9000, "limit_tokens": 8192}})
            return self.view(job)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **{f"layer_{n}": v for n, v in vectors_for(layers).items()})
        job["npz"] = buffer.getvalue()
        revision = "b" * 40 if self.mode == "wrong_revision" else self.config["revision"]
        job.update(status="succeeded", progress=None, result={
            "model": self.config["model_id"], "revision": revision, "precision": "bfloat16", "quantization": None,
            "text": self.chat_output, "rendered_prompt": "<bos>rendered", "input_token_ids": [1, 2, 3],
            "output_token_ids": [4, 5, 6], "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
            "ended_with_eos": True, "hit_token_limit": False, "outcome": "unlabeled", "label_source": None,
            "activations": {"available": True, "sha256": hashlib.sha256(job["npz"]).hexdigest(), "layers": layers,
                            "dimensions": DIMS, "token_index": 2, "token_id": 3, "token_text": "\n",
                            "capture_execution": "generation_prefill", "module_path": "model.language_model.layers"},
            "snapshot_preview": previews,
            "recording_metadata": {"chat_template": "TEMPLATE", "model_config": {"hidden_size": DIMS},
                                   "generation_config": {"do_sample": False}, "attn_implementation": "sdpa"},
            "education_trace": {**trace, "candidates": candidates},
            "packages": {"transformers": "4.57.6", "torch": "2.8.0+cu128"},
            "metrics": {"inference_memory": {"gpu_name": "NVIDIA A40"}}})
        return self.view(job)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def api():
    server = FakeAPI()
    yield server
    server.close()


def service_for(tmp_path, url, code=CODE):
    return ChatService(CONFIG, tmp_path, lambda c: RemoteAdapter(c, {"api_url": url, "access_code": code}), auto_label=False)


def finish(service):
    deadline = time.monotonic() + 30
    while service.job["status"] == "running":
        assert time.monotonic() < deadline, "Background job did not finish"
        time.sleep(.02)
    return service.job


def test_cloud_turns_are_recorded_locally_in_the_usual_format(tmp_path, api, monkeypatch):
    monkeypatch.setattr("avoidance.remote.POLL_SECONDS", 0.01)
    service = service_for(tmp_path, api.url)
    sid = service.create("Keep it short.")
    service.submit(sid, "hello")
    job = finish(service)
    assert job["status"] == "complete", job.get("error")
    kinds = [e["kind"] for e in job["events"]]
    assert kinds.index("capturing") < kinds.index("snapshot") < kinds.index("generating") < kinds.index("saving")
    assert [c["step"] for c in job["education"]["candidates"]] == [0, 1, 2]  # Live window, then final fill.
    assert job["response"] == "Hello there."
    request = api.requests[0]
    assert request["record_layers"] == [12, 24, 31, 41] and request["trace"] is True
    assert request["messages"][0] == {"role": "system", "content": "Keep it short."}
    directory = tmp_path / sid / service.sessions[sid]["latest"]
    manifest, rows, arrays = load_dataset(directory)
    for layer, vector in vectors_for(CONFIG["layers"]).items():
        np.testing.assert_array_equal(arrays[f"layer_{layer}"][0], vector)
    assert rows[0]["model_id"] == "google/gemma-3-12b-it" and rows[0]["remote"]["job_id"] == "0" * 32
    assert rows[0]["token_text"] == "\n" and rows[0]["capture_execution"] == "generation_prefill"
    assert manifest["execution"] == "remote_gpu" and manifest["hardware"]["gpu"] == "NVIDIA A40"
    assert manifest["remote_packages"]["transformers"] == "4.57.6"
    assert (directory / "chat_template.txt").read_text(encoding="utf-8") == "TEMPLATE"
    assert service.state(sid)["mode"] == "remote"
    service.submit(sid, "and more")
    assert finish(service)["status"] == "complete"
    assert api.requests[1]["messages"][-2:] == [{"role": "assistant", "content": "Hello there."},
                                                {"role": "user", "content": "and more"}]
    assert load_dataset(tmp_path / sid / service.sessions[sid]["latest"])[2]["layer_12"].shape == (2, DIMS)


def test_stop_cancels_on_the_gpu_and_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("avoidance.remote.POLL_SECONDS", 0.01)
    api = FakeAPI("hang")
    try:
        service = service_for(tmp_path, api.url)
        sid = service.create()
        service.submit(sid, "hello")
        deadline = time.monotonic() + 10
        while service.job["stage"] != "generating" and service.job["status"] == "running":
            assert time.monotonic() < deadline
            time.sleep(.01)
        service.stop()
        assert finish(service)["status"] == "stopped"
        assert api.cancelled == ["0" * 32] and service.sessions[sid]["latest"] is None
    finally:
        api.close()


@pytest.mark.parametrize("mode, message", [
    ("too_long", "reached the recording length limit"),
    ("wrong_revision", "different model checkpoint"),
])
def test_cloud_failures_keep_previous_recordings(tmp_path, monkeypatch, mode, message):
    monkeypatch.setattr("avoidance.remote.POLL_SECONDS", 0.01)
    api = FakeAPI(mode)
    try:
        service = service_for(tmp_path, api.url)
        sid = service.create()
        service.submit(sid, "hello")
        job = finish(service)
        assert job["status"] == "error" and message in job["error"]
        assert service.sessions[sid]["latest"] is None
    finally:
        api.close()


def test_offline_gpu_wrong_code_and_long_replies_are_explained(tmp_path, api):
    service = service_for(tmp_path, "http://127.0.0.1:9")  # Nothing listens on the discard port.
    sid = service.create()
    service.submit(sid, "hello")
    assert "Can't reach the cloud GPU" in finish(service)["error"]
    service = service_for(tmp_path / "b", api.url, code="wrong-code-wrong-code-wrong-code")
    sid = service.create()
    service.submit(sid, "hello")
    assert "did not accept the access code" in finish(service)["error"]
    service = service_for(tmp_path / "c", api.url)
    sid = service.create()
    service.submit(sid, "hello", reply_limit=2048)
    assert "caps replies at 1,024 tokens" in finish(service)["error"]
    assert api.requests == []


def test_connection_file_reads_code_from_a_separate_file(tmp_path):
    (tmp_path / "code.txt").write_text("abc\n", encoding="utf-8")
    path = tmp_path / "gpu-connection.json"
    path.write_text(json.dumps({"api_url": "https://pod-8000.proxy.runpod.net/", "access_code_file": "code.txt"}))
    assert load_connection(path) == {"api_url": "https://pod-8000.proxy.runpod.net", "access_code": "abc"}
    path.write_text(json.dumps({"api_url": "http://example.com", "access_code": "abc"}))
    with pytest.raises(RuntimeError, match="https://"):
        load_connection(path)
    with pytest.raises(RuntimeError, match="not set up"):
        load_connection(tmp_path / "missing.json")
