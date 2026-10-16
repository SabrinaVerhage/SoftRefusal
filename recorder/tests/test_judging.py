import io
import json
import threading
import time
import zipfile

import pytest

from avoidance.chat import ChatService
from avoidance.io import load_dataset, read_jsonl
from avoidance.judging import judge_reply, parse_judgment
from avoidance.labels import label_records
from avoidance.remote import RemoteAdapter
from test_remote import FakeAPI, CONFIG, CODE, finish


@pytest.mark.parametrize('text', ['{"outcome":"safe","reason":"safe"}', 'I refuse to classify.',
                                  '{"outcome":"refused"}', '{"outcome":"answered","reason":5}'])
def test_invalid_output_is_not_a_refusal_label(text):
    with pytest.raises((ValueError, TypeError)):
        parse_judgment(text)


def test_separate_judgment_preserves_recording_and_manual_corrections(tmp_path, monkeypatch):
    monkeypatch.setattr('avoidance.remote.POLL_SECONDS', .01)
    api = FakeAPI()
    service = ChatService(CONFIG, tmp_path, lambda c: RemoteAdapter(c, {'api_url':api.url,'access_code':CODE}))
    try:
        sid = service.create()
        service.submit(sid, 'hello')
        assert finish(service)['status'] == 'complete'
        rows = service.state(sid)['rows']
        assert rows[0]['label'] == 'answered' and rows[0]['label_source'] == 'model_judge'
        assert len(api.requests) == 2
        judge = api.requests[1]
        assert 'record_layers' not in judge and 'after_job_id' not in judge and judge['trace'] is False
        evidence = json.loads(judge['messages'][1]['content'])
        assert evidence['conversation'] == [{'role':'user','content':'hello'}]
        assert evidence['assistant_reply'] == 'Hello there.'
        checkpoint = tmp_path / sid / service.sessions[sid]['latest']
        before = (checkpoint / 'activations.npz').read_bytes()
        pid = rows[0]['point_id']
        service.label(sid, pid, 'refused')
        saved_row = load_dataset(checkpoint)[1][0]
        service._label_saved(sid, saved_row)  # A late automatic result cannot erase a human correction.
        assert service.state(sid)['rows'][0]['label'] == 'refused'
        assert service.state(sid)['rows'][0]['label_source'] == 'human'
        service.label(sid, pid, 'auto')
        assert service.state(sid)['rows'][0]['label'] == 'answered'
        assert (checkpoint / 'activations.npz').read_bytes() == before
        with zipfile.ZipFile(io.BytesIO(service.export(sid))) as archive:
            assert {'auto_labels.jsonl','labels.jsonl'} <= set(archive.namelist())
        api.judge_output = 'not JSON'
        service.submit(sid, 'and more')
        assert finish(service)['status'] == 'complete'
        assert len(service.state(sid)['rows']) == 2
        assert service.state(sid)['rows'][1]['label'] == 'unlabeled'
        assert service.state(sid)['rows'][1]['label_status'] == 'error'
        assert api.requests[-2]['messages'] == [{'role':'user','content':'hello'},
                    {'role':'assistant','content':'Hello there.'},{'role':'user','content':'and more'}]
        assert len(load_dataset(tmp_path / sid / service.sessions[sid]['latest'])[1]) == 2
    finally:
        api.close()


def test_backfill_and_cancellation_keep_saved_turns(tmp_path, monkeypatch):
    monkeypatch.setattr('avoidance.remote.POLL_SECONDS', .01)
    api = FakeAPI()
    service = ChatService(CONFIG, tmp_path, lambda c: RemoteAdapter(c, {'api_url':api.url,'access_code':CODE}), auto_label=False)
    try:
        sid = service.create()
        service.submit(sid, 'hello')
        assert finish(service)['status'] == 'complete'
        service.auto_label_enabled = True
        service.auto_label_saved(sid)
        deadline = time.monotonic()+10
        while service.busy():
            assert time.monotonic()<deadline
            time.sleep(.01)
        assert service.state(sid)['rows'][0]['label_source'] == 'model_judge'
        assert len(service.state(sid)['rows']) == 1
        service.cancel.set()
        row = service._rows(sid)[0][0]
        verdict = judge_reply(service.adapter, row, service.cancel)
        assert verdict['status'] == 'error' and verdict['outcome'] == 'unlabeled'
    finally:
        api.close()


def test_saved_checkpoint_precedes_tagger_and_tagger_exception_is_nonfatal(tmp_path, monkeypatch):
    monkeypatch.setattr('avoidance.remote.POLL_SECONDS', .01)
    api = FakeAPI()
    service = ChatService(CONFIG, tmp_path, lambda c: RemoteAdapter(c, {'api_url':api.url,'access_code':CODE}))
    sid = service.create()
    def fail(adapter, row, cancel):
        assert service.sessions[sid]['latest'] is not None
        assert len(service._rows(sid)[0]) == 1
        raise OSError('Fixture annotation write failure')
    monkeypatch.setattr('avoidance.judging.judge_reply', fail)
    try:
        service.submit(sid, 'hello')
        assert finish(service)['status'] == 'complete'
        assert len(service.state(sid)['rows']) == 1
    finally:
        api.close()


def test_input_limit_does_not_truncate_or_submit():
    class Adapter:
        config = CONFIG
        host = 'fixture'
        def _request(self, *args, **kwargs):
            pytest.fail('Over-limit evidence must not be sent')
    result = judge_reply(Adapter(), {'messages':[{'role':'user','content':'x'*13000}], 'response':'text'})
    assert result['status'] == 'error' and result['outcome'] == 'unlabeled'


def test_failed_and_capped_judgments_never_become_refusal():
    class Adapter:
        config = CONFIG
        host = 'fixture'
        def _request(self, method, path, *args, **kwargs):
            return 202, {'job_id':'fixture','status':'succeeded','result':{
                'model':CONFIG['model_id'],'revision':CONFIG['revision'],
                'text':'{"outcome":"refused","reason":"fixture"}', 'hit_token_limit':True}}
    result=judge_reply(Adapter(), {'messages':[{'role':'user','content':'Hello'}],'response':'Hello'})
    assert result['status']=='error' and result['outcome']=='unlabeled'


def test_rate_limit_retry_keeps_same_request_identity(monkeypatch):
    monkeypatch.setattr('avoidance.judging.time.sleep', lambda seconds: None)
    requests = []
    class Adapter:
        config = CONFIG
        host = 'fixture'
        def _request(self, method, path, body=None, headers=None, **kwargs):
            requests.append((method, path, body, dict(headers or {})))
            if len(requests) == 1:
                return 429, {'detail': 'rate limit'}
            return 202, {'job_id': 'fixture', 'status': 'succeeded', 'result': {
                'model': CONFIG['model_id'], 'revision': CONFIG['revision'],
                'text': '{"outcome":"answered","reason":"Replies to the greeting."}'}}
    result = judge_reply(Adapter(), {'messages': [{'role': 'user', 'content': 'Hello'}],
                                    'response': 'Hello'})
    assert result['status'] == 'complete' and result['outcome'] == 'answered'
    assert len(requests) == 2 and requests[0] == requests[1]
