"""CPU-only contract tests; optional saved native tokenizer, no weights/GPU."""
import asyncio
import base64
import copy
import io
import json
import os
import threading
from unittest.mock import patch

import pytest
from SessionProxy import memory as m
from SessionProxy.test_live import native, fake_encode, ACTION


class FixtureTokenizer:
    apply_chat_template = staticmethod(native)
    encode = staticmethod(lambda s, add_special_tokens=False: fake_encode(s))


T = FixtureTokenizer()


def step(state=None, executed=None, now=1, snapshot=None, budget=30000, tokenizer=T):
    return m.action_step(state=state or {}, frame='frame'+str(now), executed=executed,
                         now=now, protocol='tiered-v1', max_input_tokens=budget,
                         tokenizer=tokenizer, memory_snapshot=snapshot)


def ack(p):
    return dict(frame=p['state']['records'][-1]['image'], action=ACTION,
                note='uncertain location')


def test_prefix_immutable_and_hardcap_never_compacts():
    p=step()
    before=copy.deepcopy(p)
    with patch.object(m, 'thumbnail', side_effect=AssertionError), patch.object(m, 'prepare', side_effect=AssertionError):
        q=step(p['state'], ack(p), 2)
        assert q['messages'][:len(p['messages'])]==p['messages']
        old=m._encode(T,m.render_text(m._native(p['messages']),native,p['state']['counts']))
        new=m._encode(T,m.render_text(m._native(q['messages']),native,q['state']['counts']))
        assert new[:len(old)]==old
        with pytest.raises(ValueError,match='hard token cap'):
            step(q['state'], ack(q), 3, budget=1000)
    assert p==before


@pytest.mark.asyncio
async def test_snapshot_suffix_no_loss_and_pending_excluded():
    p=step()
    p=step(p['state'],ack(p),2)
    initial=copy.deepcopy(p['state'])
    async def summarize(messages):
        assert 'frame2' not in json.dumps(messages)
        assert ACTION in json.dumps(messages)
        return 'Observed movement, destination uncertain.'
    snap=await m.compact_state(state=p['state'],now=3,summarize=summarize,tokenizer=T)
    assert p['state']==initial
    assert snap['through']==1
    assert snap['messages'][-1]['role']=='assistant'
    p=step(p['state'],ack(p),3)
    q=step(p['state'],ack(p),4,snapshot=json.dumps(snap).encode())
    assert [r['id'] for r in q['state']['records']]==['1','2','3','4']
    assert [r.get('action') for r in q['state']['records']]==[ACTION,ACTION,ACTION,'']
    assert sum(x['role']=='assistant' for x in q['messages'])==3
    r=step(q['state'],ack(q),5,snapshot=json.dumps(snap).encode())
    assert not r['compacted']
    assert r['messages'][:len(q['messages'])]==q['messages']


def test_dryrun_never_invents_actions():
    p=step()
    q=step(p['state'],None,2)
    assert q['compacted']
    assert not any(x['role']=='assistant' for x in q['messages'])
    assert not any(r['action'] for r in q['state']['records'])


@pytest.mark.asyncio
async def test_cutoff_and_no_new_ack_skip_model():
    p=step()
    p=step(p['state'],ack(p),2)
    called=[]
    async def summarize(msg):
        called.append(msg)
        return 'No definite destination.'
    assert await m.compact_state(state=p['state'],now=1,summarize=summarize,tokenizer=T) is None
    assert not called
    snap=await m.compact_state(state=p['state'],now=2,summarize=summarize,tokenizer=T)
    assert snap['through']==1
    p['state']['snapshot_through']=1
    assert await m.compact_state(state=p['state'],now=3,summarize=summarize,tokenizer=T) is None
    assert len(called)==1


@pytest.mark.asyncio
async def test_thumbnail_cpu_thread_and_genuine_notes():
    p=step()
    for i in range(2,22):
        p=step(p['state'],ack(p),i)
    event_thread=threading.get_ident()
    calls=[]
    def thumb(url):
        calls.append(threading.get_ident())
        return 'thumb-'+url
    async def summarize(messages):
        assert threading.get_ident()==event_thread
        assert any('thumb-' in json.dumps(x) for x in messages)
        await asyncio.sleep(0)
        return 'The objective remains uncertain.'
    with patch.object(m,'thumbnail',thumb):
        snap=await m.compact_state(state=p['state'],now=22,summarize=summarize,tokenizer=T)
    assert calls and all(x!=event_thread for x in calls)
    assert snap['notes'][0]['provenance']=='model-generated-unreviewed'
    assert 'reviewed' not in snap['notes'][0]
    assert snap['counts'].count(920)==16
    assert 60 in snap['counts']
    assert all(int(r['id'])<=snap['through'] for r in snap['records'])


def test_real_thumbnail_grid_padding():
    from PIL import Image
    out=io.BytesIO()
    Image.new('RGB',(1280,736),'white').save(out,format='PNG')
    url=m.thumbnail('data:image/png;base64,'+base64.b64encode(out.getvalue()).decode())
    im=Image.open(io.BytesIO(base64.b64decode(url.split(',')[1])))
    assert im.size==(320,192)
    assert im.crop((0,180,320,192)).getbbox() is None


@pytest.mark.asyncio
async def test_invalid_model_output_rejected():
    p=step();p=step(p['state'],ack(p),2)
    for value in ('',None,'<|memory|>bad', 'x'*4000):
        async def summarize(messages):
            return value
        with pytest.raises(ValueError):
            await m.compact_state(state=p['state'],now=3,summarize=summarize,tokenizer=T)


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get('LUMINE_TEST_PROCESSOR'),reason='saved native tokenizer not supplied')
async def test_saved_native_tokenizer_snapshots():
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(os.environ['LUMINE_TEST_PROCESSOR'],local_files_only=True,trust_remote_code=False)
    p=step(tokenizer=t)
    q=step(p['state'],ack(p),2,tokenizer=t)
    old=m._encode(t,m.render_text(m._native(p['messages']),t.apply_chat_template,p['state']['counts']))
    new=m._encode(t,m.render_text(m._native(q['messages']),t.apply_chat_template,q['state']['counts']))
    assert new[:len(old)]==old
    for i in range(3,21):
        q=step(q['state'],ack(q),i,tokenizer=t)
    async def summarize(messages):
        return 'Objective uncertain.'
    with patch.object(m,'thumbnail',lambda url:'thumbnail'):
        snap=await m.compact_state(state=q['state'],now=21,summarize=summarize,tokenizer=t)
    assert 60 in snap['counts'] and snap['counts'].count(920)==16
    r=step(q['state'],ack(q),21,snapshot=json.dumps(snap).encode(),tokenizer=t)
    assert r['input_tokens']==m.count(t,r['messages'],r['state']['counts'])


@pytest.mark.asyncio
async def test_blocked_thumbnail_and_model_do_not_block_actions():
    p=step()
    for i in range(2,20):
        p=step(p['state'],ack(p),i)
    entered=asyncio.Event()
    loop=asyncio.get_running_loop()
    release=threading.Event()
    model_entered=asyncio.Event()
    model_release=asyncio.Event()
    def thumb(url):
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(3):
            raise TimeoutError('test worker release')
        return 'thumb'
    async def summarize(messages):
        model_entered.set()
        await model_release.wait()
        return 'Uncertain objective.'
    with patch.object(m,'thumbnail',thumb):
        task=asyncio.create_task(m.compact_state(state=p['state'],now=20,summarize=summarize,tokenizer=T))
        try:
            await asyncio.wait_for(entered.wait(),2)
            q=step(p['state'],ack(p),20)
            assert q['state']['sequence']==20
            assert not task.done()
            release.set()
            await asyncio.wait_for(model_entered.wait(),2)
            r=step(q['state'],ack(q),21)
            assert r['state']['sequence']==21
            assert not task.done()
            model_release.set()
            snap=await asyncio.wait_for(task,2)
            assert snap['through']==18
        finally:
            release.set()
            model_release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
async def test_http_real_bundled_hook_blocked_cpu_then_model(monkeypatch):
    import secrets
    from PIL import Image
    from aiohttp.test_utils import TestClient, TestServer
    from SessionProxy import adapter
    from SessionProxy.service import Config, create_app
    monkeypatch.setattr(adapter, '_tokenizer', lambda: T)
    output=io.BytesIO()
    Image.new('RGB',(1280,736)).save(output,format='PNG')
    frame='data:image/png;base64,'+base64.b64encode(output.getvalue()).decode()
    clock=[1.]
    key=secrets.token_urlsafe(40)
    app=create_app(Config(key,'fixture',protocol='tiered-v1'),adapter.build_step,clock=lambda:clock[0])
    proxy=app['proxy']
    async def infer(messages):
        return ACTION
    proxy.infer=infer
    entered=asyncio.Event()
    loop=asyncio.get_running_loop()
    release=threading.Event()
    model_entered=asyncio.Event()
    model_release=asyncio.Event()
    def thumb(url):
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            raise TimeoutError('thumbnail test release')
        return frame
    async def summarize(messages):
        model_entered.set()
        await model_release.wait()
        return 'Destination uncertain.'
    proxy.infer_memory=summarize
    monkeypatch.setattr(m,'thumbnail',thumb)
    body=dict(session_id=secrets.token_hex(16),request_id='',frame=frame,
              last_executed_action=None,executed_action_id=None)
    async with TestClient(TestServer(app)) as client:
        # Populate a real ACK history before starting this one controlled job.
        proxy.memory.closed=True
        async def http_step():
            clock[0]+=1
            body['request_id']=secrets.token_hex(16)
            response=await asyncio.wait_for(client.post('/v1/session/step',json=body,
                headers={'Authorization':'Bearer '+key}),1)
            assert response.status==200,await response.text()
            value=await response.json()
            body.update(last_executed_action=value['action'],executed_action_id=value['action_id'])
        for _ in range(20):
            await http_step()
        proxy.memory.closed=False
        proxy.memory.submit(body['session_id'],proxy.sessions[body['session_id']])
        try:
            await asyncio.wait_for(entered.wait(),2)
            await http_step()  # HTTP inference finishes while real hook CPU is blocked.
            assert not model_entered.is_set()
            release.set()
            await asyncio.wait_for(model_entered.wait(),2)
            await http_step()  # HTTP inference finishes while genuine summary await blocks.
            assert proxy.sessions[body['session_id']].memory_snapshot is None
            model_release.set()
            async with asyncio.timeout(2):
                while proxy.sessions[body['session_id']].memory_snapshot is None:
                    await asyncio.sleep(.001)
            published=json.loads(proxy.sessions[body['session_id']].memory_snapshot)
            assert published['through']>=19
            await http_step()
            rows=proxy.sessions[body['session_id']].state['records']
            assert rows[-1]['id']=='23'
            assert all(r['action']==ACTION for r in rows[-4:-1])
            assert not rows[-1]['action']
        finally:
            release.set()
            model_release.set()
