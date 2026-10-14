import json

import pytest

from avoidance.chat import ChatService
from avoidance.mapping import export_map
from test_chat import settings, wait_for_job
from test_pipeline import tiny_adapter  # noqa: F401  (pytest fixture)


def record(service, turns):
    sid = service.create()
    for prompt in turns:
        service.submit(sid, prompt)
        assert wait_for_job(service)["status"] == "complete"
    return sid


def test_map_combines_conversations_per_layer(tmp_path, tiny_adapter):
    runs = tmp_path / "runs"
    service = ChatService(settings(tiny_adapter), runs, lambda _: tiny_adapter)
    first = record(service, ["hello", "hello world", "world"])
    second = record(service, ["world hello", "hello hello", "world world", "hello"])
    service.create()  # A conversation without completed turns is skipped.
    message = export_map(runs, tmp_path / "map" / "recordings.js", neighbors=4)
    assert "7 turns from 2 conversations" in message
    data = json.loads((tmp_path / "map" / "recordings.json").read_text(encoding="utf-8"))
    script = (tmp_path / "map" / "recordings.js").read_text(encoding="utf-8")
    assert script.startswith("window.RECORDINGS=") and json.loads(script[18:-2]) == data
    assert data["kind"] == "recordings" and data["synthetic"] is False and data["layers"] == [0, 1]
    points = data["points"]
    assert [p["conversation_id"] for p in points] == [first] * 3 + [second] * 4
    assert [p["previous_point_id"] for p in points[3:5]] == [None, points[3]["point_id"]]
    assert all(set(p["layers"]) == {"0", "1"} and [p["x"], p["y"], p["z"]] == p["layers"][str(data["default_layer"])]
               for p in points)
    assert points[1]["prompt"] == "hello world" and isinstance(points[1]["response"], str)
    with pytest.raises(ValueError, match="Layer 5 was not recorded"):
        export_map(runs, tmp_path / "other.js", layer=5, neighbors=4)


def test_labels_place_turns_on_the_donut(tmp_path, tiny_adapter):
    runs = tmp_path / "runs"
    service = ChatService(settings(tiny_adapter), runs, lambda _: tiny_adapter)
    sid = record(service, ["hello", "hello world", "world", "world hello", "hello hello", "world world"])
    message = export_map(runs, tmp_path / "a.js", neighbors=4)
    assert "no donut yet" in message
    assert "unavailable" in json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))["semantic"]
    rows = service.state(sid)["rows"]
    with pytest.raises(ValueError, match="Choose answered"):
        service.label(sid, rows[0]["point_id"], "maybe")
    with pytest.raises(ValueError, match="not in this conversation"):
        service.label(sid, "nope", "refused")
    service.label(sid, rows[0]["point_id"], "refused")  # Changed later: the latest label wins.
    for row, outcome in zip(rows, ["answered", "refused", "answered", "refused", "redirected"]):
        service.label(sid, row["point_id"], outcome)
    assert [r["label"] for r in service.state(sid)["rows"]] == ["answered", "refused", "answered", "refused", "redirected", "unlabeled"]
    assert len((runs / sid / "labels.jsonl").read_text(encoding="utf-8").splitlines()) == 6  # Every change kept.
    message = export_map(runs, tmp_path / "b.js", neighbors=4)
    assert "donut from 2 refused, 1 redirected, 2 answered" in message
    data = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert data["semantic"]["core_label"] == "refused" and data["semantic"]["labels"]["unlabeled"] == 1
    scores = {p["outcome"]: [] for p in data["points"]}
    for p in data["points"]:
        assert all(k in p for k in ("core_score", "angle", "lift")) and -1 <= p["lift"] <= 1
        scores[p["outcome"]].append(p["core_score"])
    # By construction the labelled groups average 0 (answered) and 1 (refused).
    assert abs(sum(scores["answered"]) / 2) < 1e-5 and abs(sum(scores["refused"]) / 2 - 1) < 1e-5


def test_map_refuses_mixed_recording_settings(tmp_path, tiny_adapter):
    runs = tmp_path / "runs"
    record(ChatService(settings(tiny_adapter), runs, lambda _: tiny_adapter), ["hello"] * 3)
    other = dict(settings(tiny_adapter), revision="b" * 40)
    record(ChatService(other, runs, lambda _: tiny_adapter), ["world"] * 3)
    with pytest.raises(ValueError, match="different settings"):
        export_map(runs, tmp_path / "map.js")
