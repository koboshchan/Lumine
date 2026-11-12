import base64
import io
import sys
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import create_app, request_sample, encode_prompt
from train import ActionCollator

ACTION = '<|action_start|>0 0 0; W; W; W; W; W; W<|action_end|>'


def frame(size=(1280, 736), padding=False):
    im = Image.new('RGB', size)
    im.paste('green', (0, 0, size[0], min(size[1], 720)))
    if padding:
        im.putpixel((0, 735), (255, 0, 0))
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def body(url=None):
    return dict(model='lumine-mini-2b', temperature=0, max_tokens=192, stream=False,
                messages=[{'role': 'user', 'content': [{'type': 'image_url',
                           'image_url': {'url': url or frame()}}]}])


class Fake:
    config = {'recent_frames': 8}
    def __init__(self):
        self.samples = []
    def complete(self, sample, tokens):
        self.samples.append(sample)
        return ACTION, 32, 25


def test_transport():
    sample = request_sample(body())
    assert sample['current']['image'].size == (640, 360)
    assert sample['recent'] == []
    with pytest.raises(ValueError):
        request_sample(body(frame(padding=True)))
    with pytest.raises(ValueError):
        request_sample(body('https://example.com/x.png'))


def test_http_and_execution_history():
    fake = Fake()
    client = TestClient(create_app(fake, 'secret'))
    headers = {'Authorization': 'Bearer secret'}
    assert client.post('/v1/chat/completions', json=body()).status_code == 401
    response = client.post('/v1/chat/completions', json=body(), headers=headers)
    assert response.status_code == 200
    assert response.json()['choices'][0]['message']['content'] == ACTION
    b = dict(session_id='a'*32, request_id='b'*32, frame=frame(),
             last_executed_action=None, executed_action_id=None)
    r = client.post('/v1/session/step', json=b, headers=headers)
    assert r.status_code == 200, r.text
    action_id = r.json()['action_id']
    assert not fake.samples[-1]['recent']
    assert client.post('/v1/session/step', json=b, headers=headers).status_code == 409
    b.update(request_id='c'*32, last_executed_action=ACTION, executed_action_id='d'*32)
    assert client.post('/v1/session/step', json=b, headers=headers).status_code == 409
    b['executed_action_id'] = action_id
    r = client.post('/v1/session/step', json=b, headers=headers)
    assert r.status_code == 200, r.text
    assert fake.samples[-1]['recent'][0]['action'] == ACTION
    b.update(request_id='d'*32, last_executed_action=None, executed_action_id=None)
    assert client.post('/v1/session/step', json=b, headers=headers).status_code == 200
    assert len(fake.samples[-1]['recent']) == 1  # Dry-run prediction never acknowledged.
    assert client.post('/v1/session/reset', json={'session_id': 'a'*32}, headers=headers).status_code == 200
    assert client.post('/v1/session/step', json=b, headers=headers).status_code == 409


def test_native_prompt_parity():
    import json
    import torch
    from transformers import AutoProcessor
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'config.json').read_text())
    p = AutoProcessor.from_pretrained(config['model'], revision=config['model_revision'], trust_remote_code=True)
    p.chat_template = (root/'chat_template.jinja').read_text()
    sample = request_sample(body())
    sample['recent'] = [dict(sample['current'], action=ACTION, timestamp_s=.1)]
    sample['current']['timestamp_s'] = .2
    expected = p.apply_chat_template(ActionCollator(p, config).messages(sample)[:-1],
        tokenize=True, return_dict=True, return_tensors='pt', add_generation_prompt=True,
        enable_thinking=False)
    actual = encode_prompt(p, config, sample)
    assert expected.keys() == actual.keys()
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key
    rendered = p.apply_chat_template(ActionCollator(p, config).messages(sample)[:-1], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    assert 'Previous action: ' + ACTION in rendered
    assert '<think>' not in rendered


def test_bootstrap_dryrun_only():
    fake = Fake()
    client = TestClient(create_app(fake, 'secret'))
    headers = {'Authorization': 'Bearer secret'}
    sid = 'e' * 32
    b = dict(session_id=sid, frames=[frame(), frame()], timestamps_s=[.0065, .3909],
             acknowledge_no_inputs=True)
    denied = dict(b, acknowledge_no_inputs=False)
    assert client.post('/v1/session/bootstrap', json=denied, headers=headers).status_code == 400
    denied = dict(b, timestamps_s=[1, 0])
    assert client.post('/v1/session/bootstrap', json=denied, headers=headers).status_code == 400
    assert client.post('/v1/session/bootstrap', json=b, headers=headers).status_code == 200
    assert client.post('/v1/session/bootstrap', json=b, headers=headers).status_code == 409
    step = dict(session_id=sid, request_id='f'*32, frame=frame(),
                last_executed_action=None, executed_action_id=None)
    result = client.post('/v1/session/step', json=step, headers=headers)
    assert result.status_code == 200, result.text
    sample = fake.samples[-1]
    noop = '<|action_start|>0 0 0 ;  ;  ;  ;  ;  ; <|action_end|>'
    assert [row['action'] for row in sample['recent']] == [noop, noop]
    assert abs(sample['recent'][1]['timestamp_s'] - .3844) < 1e-6
    step.update(request_id='1'*32, last_executed_action=ACTION,
                executed_action_id=result.json()['action_id'])
    assert client.post('/v1/session/step', json=step, headers=headers).status_code == 409
