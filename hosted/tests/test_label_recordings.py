import json
from unittest.mock import Mock

import pytest
from hosted import label_recordings as labeling


@pytest.fixture
def harness(monkeypatch):
    action = Mock(return_value={"active": 0})
    monkeypatch.setattr(labeling, 'gateway_action', action)
    monkeypatch.setattr(labeling, 'load_connection', lambda _: {})
    monkeypatch.setattr(labeling, 'RemoteAdapter', lambda *args: object())
    monkeypatch.setattr(labeling.time, 'sleep', lambda _: None)
    monkeypatch.setattr(labeling, 'CASES', [('fixture', 'Hello', 'Hello', 'answered')])
    return action


def test_failed_validation_resumes_without_backfill(tmp_path, monkeypatch, harness):
    monkeypatch.setattr(labeling, 'judge_reply', lambda *args, **kwargs: {'status':'complete','outcome':'refused'})
    with pytest.raises(RuntimeError, match='validation failed'):
        labeling.run(tmp_path, 'unused', backfill=True)
    assert [call.args[0] for call in harness.call_args_list] == ['drain', 'resume']
    assert not json.loads((tmp_path / 'judge-validation.json').read_text())['passed']
    assert not list(tmp_path.glob('recordings/*/auto_labels.jsonl'))


def test_active_visitors_resume_without_judging(tmp_path, monkeypatch, harness):
    harness.return_value = {'active': 1}
    judge = Mock(); monkeypatch.setattr(labeling, 'judge_reply', judge)
    with pytest.raises(RuntimeError, match='Visitors are recording'):
        labeling.run(tmp_path, 'unused')
    judge.assert_not_called()
    assert harness.call_args.args == ('resume',)


def test_backfill_preserves_human_labels_and_checkpoint(tmp_path, monkeypatch, harness):
    directory = tmp_path / 'recordings/session'; directory.mkdir(parents=True)
    (directory / 'session.json').write_text(json.dumps({'latest':'checkpoint'}))
    human = json.dumps({'point_id':'human','outcome':'refused'})
    (directory / 'labels.jsonl').write_text(human)
    checkpoint = directory / 'checkpoint'; checkpoint.write_bytes(b'immutable fixture')
    monkeypatch.setattr(labeling, 'load_dataset', lambda _: ({}, [{'point_id':'human'}, {'point_id':'missing'}], {}))
    monkeypatch.setattr(labeling, 'judge_reply', lambda *args, **kwargs: {'status':'complete','outcome':'answered'})
    report = labeling.run(tmp_path, 'unused', backfill=True)
    assert report['backfilled'] == {'answered': 1}
    assert json.loads((directory / 'auto_labels.jsonl').read_text())['point_id'] == 'missing'
    assert (directory / 'labels.jsonl').read_text() == human
    assert checkpoint.read_bytes() == b'immutable fixture'
    assert harness.call_args.args == ('resume',)
