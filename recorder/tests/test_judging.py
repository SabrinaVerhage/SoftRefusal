import io
import json
import threading
import time
import zipfile

import pytest

from avoidance.chat import ChatService
from avoidance.io import load_dataset, read_json, read_jsonl
from avoidance.judging import (JUDGE_PROMPT, JUDGE_VERSION, THINKING_JUDGE_PROMPT, THINKING_JUDGE_VERSION,
                               judge_reply, judge_setup, parse_judgment, split_thinking)
from avoidance.labels import label_records
from avoidance.remote import RemoteAdapter
from test_remote import FakeAPI, CONFIG, CODE, finish

HERMES = read_json('configs/hermes4-14b-remote.json')


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


class FixtureJudge:
    """A judge backend that answers immediately with a fixed output and logs requests."""
    def __init__(self, config, text='{"outcome":"answered","reason":"fixture"}'):
        self.config, self.host, self.text, self.requests = config, 'fixture', text, []

    def _request(self, method, path, body=None, headers=None, **kwargs):
        self.requests.append(body)
        return 202, {'job_id': 'fixture', 'status': 'succeeded', 'result': {
            'model': self.config['model_id'], 'revision': self.config['revision'], 'text': self.text}}


@pytest.mark.parametrize('text, expected', [
    ('Plain answer.', (None, 'Plain answer.')),
    ('<think>Weigh it.</think>\n\nFinal.', ('Weigh it.', 'Final.')),
    ('<think>Still going', ('Still going', '')),
    ('Weigh it.</think> Final.', ('Weigh it.', 'Final.')),
    ('Intro. <think>x</think> End.', ('x', 'Intro.  End.')),
])
def test_split_thinking_matches_room_display(text, expected):
    assert split_thinking(text) == expected


@pytest.mark.parametrize('text, outcome', [
    ('<think>The reply declines.</think>\n{"outcome":"refused","reason":"Declines."}', 'refused'),
    ('<think>\n</think>\n```json\n{"outcome":"redirected","reason":"Offers another route."}\n```', 'redirected'),
    ('It declines.</think>{"outcome":"refused","reason":"Declines."}', 'refused'),
    ('{"outcome":"answered","reason":"Explains the </think> tag."}', 'answered'),
])
def test_judge_reasoning_before_json_is_ignored(text, outcome):
    assert parse_judgment(text)['outcome'] == outcome


@pytest.mark.parametrize('text', ['<think>{"outcome":"refused","reason":"x"}', '<think>Only thoughts.</think>'])
def test_unfinished_or_answerless_judge_thinking_is_not_a_label(text):
    with pytest.raises(ValueError):
        parse_judgment(text)


def test_gemma_judge_prompt_and_evidence_are_unchanged():
    assert judge_setup(CONFIG) == (JUDGE_VERSION, JUDGE_PROMPT, 'full_reply')
    judge = FixtureJudge(CONFIG)
    result = judge_reply(judge, {'messages': [{'role': 'user', 'content': 'Show a think tag.'}],
                                 'response': '<think>literal</think> text'})
    assert result['status'] == 'complete' and result['evidence_policy'] == 'full_reply'
    assert json.loads(judge.requests[0]['messages'][1]['content'])['assistant_reply'] == '<think>literal</think> text'


def test_hermes_judge_sees_final_answers_and_tolerates_judge_thinking(tmp_path, monkeypatch):
    monkeypatch.setattr('avoidance.remote.POLL_SECONDS', .01)
    config = dict(HERMES, auto_label=True)
    api = FakeAPI(config=config)
    api.chat_output = '<think>The user greets me. I should greet back.</think>\n\nHello there.'
    api.judge_output = '<think>It greets back.</think>\n{"outcome":"answered","reason":"Greets back."}'
    service = ChatService(config, tmp_path, lambda c: RemoteAdapter(c, {'api_url':api.url,'access_code':CODE}))
    try:
        sid = service.create(HERMES['default_instruction'])
        service.submit(sid, 'hello')
        assert finish(service)['status'] == 'complete'
        row = service.state(sid)['rows'][0]
        assert row['label'] == 'answered' and row['label_source'] == 'model_judge'
        assert row['response'] == api.chat_output  # The recording keeps the raw thinking.
        judge = api.requests[1]
        assert judge['model'] == 'hermes' and judge['max_tokens'] == 160
        assert judge['messages'][0]['content'] == THINKING_JUDGE_PROMPT
        evidence = json.loads(judge['messages'][1]['content'])
        assert evidence['assistant_reply'] == 'Hello there.'
        assert evidence['conversation'] == [{'role':'system','content':HERMES['default_instruction']},
                                            {'role':'user','content':'hello'}]
        service.submit(sid, 'and more')
        assert finish(service)['status'] == 'complete'
        assert api.requests[2]['messages'][2] == {'role':'assistant','content':api.chat_output}
        evidence = json.loads(api.requests[3]['messages'][1]['content'])
        assert evidence['conversation'][2] == {'role':'assistant','content':'Hello there.'}
        records = read_jsonl(tmp_path / sid / 'auto_labels.jsonl')
        assert [r['judge_version'] for r in records] == [THINKING_JUDGE_VERSION] * 2
        assert {r['evidence_policy'] for r in records} == {'final_answer_thinking_removed'}
        assert records[0]['raw_judgment'] == api.judge_output
    finally:
        api.close()


@pytest.mark.parametrize('response, capped', [
    ('<think>Still deliberating', False),
    ('<think>Done.</think>\n', False),
    ('<think>Done.</think> Hel', True),
])
def test_hermes_reply_without_final_answer_is_unclear_without_a_gpu_job(response, capped):
    judge = FixtureJudge(HERMES)
    result = judge_reply(judge, {'messages': [{'role':'user','content':'Hi'}], 'response': response,
                                 'hit_token_limit': capped})
    assert result['status'] == 'complete' and result['outcome'] == 'unclear' and judge.requests == []


def test_hermes_thinking_does_not_count_toward_judge_input_limit():
    judge = FixtureJudge(HERMES)
    result = judge_reply(judge, {'messages': [{'role':'user','content':'Hi'},
                                              {'role':'assistant','content':'<think>' + 'x'*7000 + '</think>Hi.'},
                                              {'role':'user','content':'Again'}],
                                 'response': '<think>' + 'y'*7000 + '</think>Hello again.'})
    assert result['status'] == 'complete' and result['outcome'] == 'answered'
    assert 'xxxxxxxxxx' not in judge.requests[0]['messages'][1]['content']
