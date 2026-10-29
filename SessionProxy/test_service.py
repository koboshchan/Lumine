import asyncio
import base64
import io
import secrets
from dataclasses import replace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image
from service import Config, create_app, parse_completion, decode_frame

ACTION = "<|action_start|>0 0 0;;;;;;<|action_end|>"
SECRET = secrets.token_urlsafe(40)
AUTH = {"Authorization": "Bearer " + SECRET}
SID = "a" * 32

def frame():
    out = io.BytesIO()
    Image.new("RGB", (1280, 720)).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()

FRAME = frame()

def body(**kw):
    return dict(session_id=SID, request_id=secrets.token_hex(16), frame=FRAME,
                last_executed_action=None, executed_action_id=None, **kw)

def builder(**kw):
    old = kw["state"].get("history", [])
    messages = old + [{"role": "user", "content": kw["frame"]}]
    return {"state": {"history": messages, "ack": kw["executed"]}, "messages": messages,
            "input_tokens": 1024, "compacted": False}

@pytest.fixture
async def client():
    app = create_app(Config(SECRET, "test"), builder)
    async with TestClient(TestServer(app, handler_cancellation=True)) as cli:
        async def infer(messages):
            return ACTION
        app["proxy"].infer = infer
        yield cli

@pytest.mark.asyncio
async def test_auth_all(client):
    for path in ["/v1/session/step", "/v1/session/reset", "/unknown"]:
        assert (await client.post(path, json={})).status == 401
    assert (await client.get("/unknown", headers=AUTH)).status == 404

@pytest.mark.asyncio
async def test_ack_replay_reset(client):
    b = body()
    r = await client.post("/v1/session/step", json=b, headers=AUTH)
    assert r.status == 200
    result = await r.json()
    p = client.app["proxy"]
    assert p.sessions[SID].state["ack"] is None
    assert (await client.post("/v1/session/step", json=b, headers=AUTH)).status == 409
    bad = body()
    bad.update(last_executed_action=ACTION, executed_action_id="0"*32)
    assert (await client.post("/v1/session/step", json=bad, headers=AUTH)).status == 409
    good = body()
    good.update(last_executed_action=ACTION, executed_action_id=result["action_id"])
    assert (await client.post("/v1/session/step", json=good, headers=AUTH)).status == 200
    assert p.sessions[SID].state["ack"]["action"] == ACTION
    assert (await client.post("/v1/session/step", json=good, headers=AUTH)).status == 409
    assert (await client.post("/v1/session/reset", json={"session_id":SID}, headers=AUTH)).status == 200
    assert not p.sessions[SID].state and p.sessions[SID].pending is None
    assert (await client.post("/v1/session/step", json=b, headers=AUTH)).status == 409

@pytest.mark.asyncio
async def test_dryrun_and_atomic_error(client):
    p = client.app["proxy"]
    assert (await client.post("/v1/session/step", json=body(), headers=AUTH)).status == 200
    assert (await client.post("/v1/session/step", json=body(), headers=AUTH)).status == 200
    assert p.sessions[SID].state["ack"] is None
    before = p.sessions[SID].state.copy()
    pending = p.sessions[SID].pending.copy()
    async def fail(messages):
        raise ValueError("secret completion MUST NOT leak")
    p.infer = fail
    b = body()
    r = await client.post("/v1/session/step", json=b, headers=AUTH)
    assert r.status == 502 and "secret completion" not in await r.text()
    assert p.sessions[SID].state == before and p.sessions[SID].pending == pending
    assert (await client.post("/v1/session/step", json=b, headers=AUTH)).status == 409

@pytest.mark.asyncio
async def test_concurrent_first_and_reset(client):
    p = client.app["proxy"]
    entered, release = asyncio.Event(), asyncio.Event()
    async def wait(messages):
        entered.set()
        await release.wait()
        return ACTION
    p.infer = wait
    task = asyncio.create_task(client.post("/v1/session/step", json=body(), headers=AUTH))
    await entered.wait()
    assert (await client.post("/v1/session/step", json=body(), headers=AUTH)).status == 409
    assert (await client.post("/v1/session/reset", json={"session_id":SID}, headers=AUTH)).status == 409
    release.set()
    assert (await task).status == 200

@pytest.mark.asyncio
async def test_expiry(client):
    p = client.app["proxy"]
    now = [1.]
    p.clock = lambda: now[0]
    assert (await client.post("/v1/session/step", json=body(), headers=AUTH)).status == 200
    now[0] += p.c.ttl + 1
    p.expire()
    assert not p.sessions

@pytest.mark.asyncio
async def test_strict_body(client):
    for changes in [{"model": "evil"}, {"frame": "bad"}, {"session_id": "bad"}, {"executed_action_id": "0"*32}]:
        b = body()
        b.update(changes)
        assert (await client.post("/v1/session/step", json=b, headers=AUTH)).status in {400,409}
    assert (await client.post("/v1/session/reset", data='{"session_id":"a","session_id":"b"}',
                              headers={**AUTH,"Content-Type":"application/json"})).status == 400

@pytest.mark.asyncio
async def test_prefix_fail_closed(client):
    p = client.app["proxy"]
    p.c.protocol = "tiered-v1"
    assert (await client.post("/v1/session/step", json=body(), headers=AUTH)).status == 200
    def bad(**kw):
        return {"state": {}, "messages":[{"role":"user","content":"changed"}],
                "input_tokens":1, "compacted":False}
    p.builder = bad
    assert (await client.post("/v1/session/step", json=body(), headers=AUTH)).status == 502

@pytest.mark.asyncio
async def test_timeout_atomic(client):
    p = client.app["proxy"]
    p.c.timeout = .01
    async def wait(messages):
        await asyncio.Event().wait()
    p.infer = wait
    b = body()
    assert (await client.post("/v1/session/step", json=b, headers=AUTH)).status == 502
    assert not p.sessions[SID].state and p.sessions[SID].pending is None
    assert not p.sessions[SID].busy

@pytest.mark.asyncio
async def test_limits(client):
    p = client.app["proxy"]
    p.c.max_sessions = 1
    assert (await client.post("/v1/session/step", json=body(), headers=AUTH)).status == 200
    b = body()
    b["session_id"] = "b"*32
    assert (await client.post("/v1/session/step", json=b, headers=AUTH)).status == 503
    p.c.session_bytes = 100
    assert (await client.post("/v1/session/step", json=body(), headers=AUTH)).status == 413

def test_parser():
    assert parse_completion(ACTION)[0] == ACTION
    assert parse_completion(ACTION + "<|memory|>uncertain location</|memory|>", True)[1] == "uncertain location"
    for action in [ACTION+"bad", ACTION.replace(";;;;;;", ";Alt Tab;;;;;"),
                   ACTION.replace("0 0 0", "1 0 0"), ACTION.replace(";;;;;;", ";Ctrl Esc;;;;;"),
                   ACTION+"<|memory|>x</|memory|>"]:
        with pytest.raises(ValueError):
            parse_completion(action)

def test_config():
    for config in [Config("x", "test"), Config(SECRET,"test",upstream="http://example.com/v1/chat/completions"),
                   Config(SECRET,"test",upstream="http://127.0.0.1:1/v1/chat/completions?x=1")]:
        with pytest.raises(ValueError):
            config.validate()

def test_png():
    assert decode_frame(FRAME).startswith("data:image/png;base64,")
    out = io.BytesIO()
    Image.new("RGB",(32,32)).save(out, format="PNG")
    with pytest.raises(ValueError):
        decode_frame("data:image/png;base64,"+base64.b64encode(out.getvalue()).decode())
    with pytest.raises(ValueError):
        decode_frame("data:image/png;base64,!!!!")

@pytest.mark.asyncio
async def test_disconnected_request_atomic(client):
    p = client.app['proxy']
    entered, cancelled = asyncio.Event(), asyncio.Event()
    async def wait(messages):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    p.infer = wait
    b = body()
    task = asyncio.create_task(client.post('/v1/session/step', json=b, headers=AUTH))
    await entered.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.wait_for(cancelled.wait(), 2)
    assert p.sessions[SID].state == {} and p.sessions[SID].pending is None
    assert b['request_id'] in p.sessions[SID].requests
    assert not p.sessions[SID].busy and p.inflight == 0 and p.reserved_bytes == 0

@pytest.mark.asyncio
async def test_real_upstream_validation(client):
    hits = []
    mode = ['good']
    async def upstream(request):
        payload = await request.json()
        hits.append(payload)
        assert request.headers['Authorization'] == 'Bearer private-test'
        assert payload['skip_special_tokens'] is False
        if mode[0] == 'redirect':
            raise web.HTTPTemporaryRedirect(location='/forbidden')
        finish = 'length' if mode[0] == 'truncated' else 'stop'
        return web.json_response({'choices':[{'finish_reason':finish,'message':{'role':'assistant','content':ACTION}}]})
    app = web.Application()
    app.router.add_post('/v1/chat/completions',upstream)
    async with TestServer(app) as server:
        p = client.app['proxy']
        from service import Proxy
        p.infer = Proxy.infer.__get__(p)
        p.c.upstream = str(server.make_url('/v1/chat/completions'))
        p.c.upstream_key = 'private-test'
        assert (await client.post('/v1/session/step',json=body(),headers=AUTH)).status == 200
        for value in ['redirect','truncated']:
            mode[0] = value
            assert (await client.post('/v1/session/step',json=body(),headers=AUTH)).status == 502
        assert len(hits) == 3

@pytest.mark.asyncio
async def test_budget_compaction(client):
    p = client.app['proxy']
    p.c.protocol = 'tiered-v1'
    def compact(**kw):
        return {'state':{},'messages':[{'role':'user','content':'compact'}],
                'input_tokens':10,'compacted':True,'pre_compaction_tokens':30001}
    p.builder = compact
    assert (await client.post('/v1/session/step',json=body(),headers=AUTH)).status == 200

def test_nonblack_padding():
    out = io.BytesIO()
    Image.new('RGB',(1280,736),'white').save(out,format='PNG')
    with pytest.raises(ValueError):
        decode_frame('data:image/png;base64,'+base64.b64encode(out.getvalue()).decode())
