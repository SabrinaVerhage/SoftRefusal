import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import torch

from avoidance.chat import ChatService, make_handler
from avoidance.model import GenerationCancelled, residual_hooks
from test_chat import settings
from test_pipeline import tiny_adapter


def test_single_prefill_matches_separate_capture(tiny_adapter, monkeypatch):
    a = tiny_adapter
    messages = [{"role": "user", "content": "hello world"}]
    rendered = a.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = a.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    with torch.inference_mode(), residual_hooks(a.blocks, [0, 1], -1) as baseline:
        a.backbone(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False)
    original = a.backbone.forward
    lengths = []
    def counted(*args, **kwargs):
        lengths.append(kwargs["input_ids"].shape[1] if kwargs.get("input_ids") is not None else kwargs["inputs_embeds"].shape[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(a.backbone, "forward", counted)
    row, vectors = a.step(messages)
    assert lengths.count(inputs.input_ids.shape[1]) == 1
    for layer in baseline:
        np.testing.assert_allclose(vectors[layer], baseline[layer], atol=1e-6, rtol=1e-6)
    assert row["capture_execution"] == "generation_prefill"


def test_cancel_during_prefill_removes_hooks(tiny_adapter):
    cancel = threading.Event()
    def event(kind, **data):
        if kind == "layer_pass":
            cancel.set()
    with pytest.raises(GenerationCancelled):
        tiny_adapter.step([{"role": "user", "content": "hello"}], on_event=event, cancel_event=cancel)
    assert all(not b._forward_hooks for b in tiny_adapter.blocks)
    assert not tiny_adapter.model.get_input_embeddings()._forward_hooks


def test_shutdown_cancels_busy_job_and_waits_for_log(tmp_path, tiny_adapter):
    ready = threading.Event()
    class Slow:
        def step(self, messages, on_event, cancel_event, **kwargs):
            ready.set()
            cancel_event.wait(10)
            raise GenerationCancelled("Stopped during test")
    service = ChatService(settings(tiny_adapter), tmp_path, lambda _: Slow())
    sid = service.create()
    service.submit(sid, "hello")
    assert ready.wait(10)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/shutdown", data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            assert json.load(r)["ok"]
        thread.join(5)
        assert not thread.is_alive()
        assert service.job["status"] == "stopped"
        assert service.closing
        result = tmp_path / sid / "attempts" / service.job["id"] / "result.json"
        assert json.loads(result.read_text())["status"] == "stopped"
        assert service.sessions[sid]["latest"] is None
    finally:
        server.shutdown()
        server.server_close()
