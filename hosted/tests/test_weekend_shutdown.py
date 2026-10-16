"""Exercise shutdown guardrails without calling RunPod or stopping compute."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def timer(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "deploy/greenhost/weekend_shutdown.py"
    spec = importlib.util.spec_from_file_location("weekend_shutdown", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key = tmp_path / "key"
    key.write_text("synthetic-scoped-key")
    config = tmp_path / "config.json"
    def configure(deadline, check=False):
        config.write_text(json.dumps({"pod_id": "movffvxo1jho2z", "deadline_utc": deadline, "key_file": str(key)}))
        monkeypatch.setattr(sys, "argv", ["timer", "--config", str(config)] + (["--check"] if check else []))
    return module, configure


def test_shutdown_never_mutates_before_deadline(timer, monkeypatch):
    module, configure = timer
    calls = []
    def request(key, query):
        calls.append(query)
        return {"pod": {"id": "movffvxo1jho2z", "desiredStatus": "RUNNING"}}
    monkeypatch.setattr(module, "request", request)
    configure("2999-10-18T21:59:00+00:00")
    with pytest.raises(RuntimeError, match="before"):
        module.main()
    assert len(calls) == 1 and calls[0].startswith("query")


def test_check_is_readonly_even_after_deadline(timer, monkeypatch):
    module, configure = timer
    calls = []
    def request(key, query):
        calls.append(query)
        return {"pod": {"id": "movffvxo1jho2z", "desiredStatus": "RUNNING"}}
    monkeypatch.setattr(module, "request", request)
    configure("2000-10-18T21:59:00+00:00", check=True)
    module.main()
    assert len(calls) == 1 and calls[0].startswith("query")


def test_due_timer_only_stops_configured_pod(timer, monkeypatch):
    module, configure = timer
    calls = []
    def request(key, query):
        calls.append(query)
        if query.startswith("query"):
            return {"pod": {"id": "movffvxo1jho2z", "desiredStatus": "RUNNING"}}
        return {"podStop": {"id": "movffvxo1jho2z", "desiredStatus": "EXITED"}}
    monkeypatch.setattr(module, "request", request)
    configure("2000-10-18T21:59:00+00:00")
    module.main()
    assert len(calls) == 2
    assert 'podStop(input: {podId: "movffvxo1jho2z"})' in calls[1]


def test_already_stopped_is_idempotent(timer, monkeypatch):
    module, configure = timer
    calls = []
    def request(key, query):
        calls.append(query)
        return {"pod": {"id": "movffvxo1jho2z", "desiredStatus": "EXITED"}}
    monkeypatch.setattr(module, "request", request)
    configure("2000-10-18T21:59:00+00:00")
    module.main()
    assert len(calls) == 1
