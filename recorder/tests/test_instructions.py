import pytest

from avoidance.chat import ChatService, DEFAULT_INSTRUCTION
from avoidance.io import load_dataset
from test_chat import settings, wait_for_job
from test_pipeline import tiny_adapter


def test_instruction_persists_changes_and_preserves_old_turns(tmp_path, tiny_adapter):
    service = ChatService(settings(tiny_adapter), tmp_path, lambda _: tiny_adapter)
    sid = service.create(DEFAULT_INSTRUCTION)
    service.submit(sid, "hello")
    assert wait_for_job(service)["status"] == "complete"
    first = service.sessions[sid]["latest"]
    _, rows, _ = load_dataset(tmp_path / sid / first)
    assert rows[0]["messages"][0] == {"role": "system", "content": DEFAULT_INSTRUCTION}
    assert rows[0]["prompt"] == "hello"
    service.set_instruction(sid, "Answer briefly.")
    service.submit(sid, "world")
    assert wait_for_job(service)["status"] == "complete"
    _, rows, _ = load_dataset(tmp_path / sid / service.sessions[sid]["latest"])
    assert rows[0]["instruction"] == DEFAULT_INSTRUCTION
    assert rows[1]["instruction"] == "Answer briefly."
    assert rows[1]["messages"][0]["content"] == "Answer briefly."
    assert len(rows[1]["messages"]) == 4
    assert rows[1]["messages"][1]["content"] == "hello"
    assert load_dataset(tmp_path / sid / first)[1][0]["instruction"] == DEFAULT_INSTRUCTION
    restored = ChatService(settings(tiny_adapter), tmp_path)
    assert restored.state(sid)["instruction"] == "Answer briefly."
    restored.set_instruction(sid, "")
    assert restored.state(sid)["instruction"] == ""
    with pytest.raises(ValueError):
        restored.set_instruction(sid, None)
    restored.job = {"status": "running"}
    with pytest.raises(ValueError, match="Wait"):
        restored.set_instruction(sid, "Change while busy")
