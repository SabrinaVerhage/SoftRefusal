import numpy as np
import pytest

from avoidance.chat import ChatService
from test_chat import settings, wait_for_job
from test_pipeline import tiny_adapter


def test_observation_does_not_change_answers_or_vectors(tiny_adapter):
    a = tiny_adapter
    messages = [{"role": "user", "content": "hello world"}]
    plain, vectors = a.step(messages, max_new_tokens=3)
    events = []
    observed, observed_vectors = a.step(messages, on_event=lambda kind, **data: events.append(kind), max_new_tokens=3)
    assert observed["response_token_ids"] == plain["response_token_ids"]
    for layer in vectors:
        np.testing.assert_array_equal(vectors[layer], observed_vectors[layer])
    trace = observed["education_trace"]
    assert len(trace["layers"]) == len(a.blocks)
    assert trace["tokens"]["total"] == observed["input_token_count"]
    assert np.asarray(trace["embeddings"]["values"]).shape == (observed["input_token_count"], 16)
    assert len(trace["candidates"]) == observed["generated_token_count"]
    assert [entry["top"][0]["id"] for entry in trace["candidates"]] == observed["response_token_ids"]
    for entry in trace["candidates"]:
        assert 0 < sum(t["probability"] for t in entry["top"]) <= 1.000001
    assert observed["generation_settings"]["max_new_tokens"] == 3
    assert a.config["max_new_tokens"] == 2
    assert all(not b._forward_hooks for b in a.blocks)
    assert not a.model.get_input_embeddings()._forward_hooks
    assert events.index("embeddings") < events.index("generating") < events.index("candidates")


def test_limits_trace_persistence_and_closing(tmp_path, tiny_adapter):
    service = ChatService(settings(tiny_adapter), tmp_path, lambda _: tiny_adapter)
    sid = service.create()
    for invalid in (0, -1, 2049, True, "512"):
        with pytest.raises(ValueError, match="Reply limit"):
            service.submit(sid, "hello", invalid)
    service.submit(sid, "hello", 1)
    assert wait_for_job(service)["status"] == "complete"
    restored = ChatService(settings(tiny_adapter), tmp_path)
    row = restored.state(sid)["rows"][0]
    assert row["generation_settings"]["max_new_tokens"] == 1
    assert len(row["education_trace"]["candidates"]) == 1
    assert len(row["education_trace"]["layers"]) == 2
    service.closing = True
    with pytest.raises(ValueError, match="stopping"):
        service.submit(sid, "hello")
    with pytest.raises(ValueError, match="stopping"):
        service.create()
