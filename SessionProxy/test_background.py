"""Deterministic scheduling tests, not GPU or 5fps timing evidence."""
import asyncio
import copy
import json
import threading

import pytest
from aiohttp.test_utils import TestClient, TestServer

from service import Config, create_app
from test_service import ACTION, AUTH, SECRET, SID, body


class SnapshotBuilder:
    snapshot_mode = True

    def __init__(self, blocked):
        self.blocked = blocked
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.finished = asyncio.Event()
        self.calls = []

    def __call__(self, **kw):
        state = copy.deepcopy(kw["state"])
        rows = state.setdefault("records", [])
        if kw["executed"]:
            rows[-1]["action"] = kw["executed"]["action"]
        seq = state.get("sequence", 0) + 1
        rows.append({"id":str(seq), "image":kw["frame"], "action":""})
        state["sequence"] = seq
        self.calls.append(kw.get("memory_snapshot"))
        messages = state.setdefault("messages", [])
        messages.append({"role":"user", "content":str(seq)})
        return {"state":state,"messages":messages,"input_tokens":1024,"compacted":False}

    async def compact(self, *, state, now, summarize):
        # Both paths execute work that is deliberately blocked independently.
        # The action request MUST complete before release is signalled.
        if self.blocked == "thumbnail":
            self.entered.set()
            await asyncio.to_thread(self.release.wait)
        else:
            await summarize([{"role":"user","content":"genuine note job"}])
        self.finished.set()
        return {"through":state["sequence"]-1,"boundary_ms":int(now*1000),
                "messages":[],"notes":["fixture model text"],"records":[],"counts":[]}


async def post(client, result=None):
    b = body()
    if result:
        b.update(last_executed_action=result["action"],executed_action_id=result["action_id"])
    response = await client.post("/v1/session/step",json=b,headers=AUTH)
    assert response.status == 200, await response.text()
    return await response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked",["thumbnail","note"])
async def test_action_completes_while_memory_deliberately_blocked(blocked):
    builder = SnapshotBuilder(blocked)
    app = create_app(Config(SECRET,"test",protocol="tiered-v1",memory_interval=0),builder)
    p = app["proxy"]
    async def infer(messages): return ACTION
    async def note(messages):
        builder.entered.set()
        await asyncio.to_thread(builder.release.wait)
        return "model written note"
    p.infer, p.infer_memory = infer, note
    try:
        async with TestClient(TestServer(app)) as client:
            result = await post(client)
            result = await post(client,result)
            await asyncio.wait_for(builder.entered.wait(),2)
            # No sleep or benchmark assertion. Release is provably unset until
            # after the independent action HTTP response has completed.
            result = await asyncio.wait_for(post(client,result),2)
            assert not builder.release.is_set() and not builder.finished.is_set()
            assert p.sessions[SID].memory_snapshot is None
            assert p.memory.active is not None
            assert len(p.memory.pending) == 1
            assert all(value is None for value in builder.calls)
            builder.release.set()
            await asyncio.wait_for(builder.finished.wait(),2)
    finally:
        builder.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidate",["reset","ttl","reset-and-reuse"])
async def test_inflight_memory_cannot_resurrect_session(invalidate):
    builder = SnapshotBuilder("thumbnail")
    now = [1.]
    app = create_app(Config(SECRET,"test",protocol="tiered-v1",memory_interval=0),builder,lambda:now[0])
    p = app["proxy"]
    async def infer(messages): return ACTION
    p.infer = infer
    try:
        async with TestClient(TestServer(app)) as client:
            result = await post(client)
            result = await post(client,result)
            await asyncio.wait_for(builder.entered.wait(),2)
            await post(client,result)
            assert SID in p.memory.pending
            epoch = p.sessions[SID].epoch
            if invalidate == "ttl":
                now[0] += p.c.ttl+1
                p.expire()
                assert SID not in p.sessions
            else:
                r = await client.post("/v1/session/reset",json={"session_id":SID},headers=AUTH)
                assert r.status == 200
                assert p.sessions[SID].epoch != epoch
                if invalidate == "reset-and-reuse":
                    await post(client)
            builder.release.set()
            # finished is set before returning the candidate; FIFO event-loop
            # callback below runs after the synchronous publication decision.
            await asyncio.wait_for(builder.finished.wait(),2)
            if invalidate == "ttl":
                assert SID not in p.sessions
            else:
                assert p.sessions[SID].memory_snapshot is None
                if invalidate == "reset":
                    assert p.sessions[SID].state == {}
            assert SID not in p.memory.pending
    finally:
        builder.release.set()


@pytest.mark.asyncio
async def test_queue_coalesces_and_bounds_and_completed_snapshot_is_bytes():
    builder = SnapshotBuilder("thumbnail")
    app = create_app(Config(SECRET,"test",protocol="tiered-v1",memory_interval=0),builder)
    p = app["proxy"]
    async def infer(messages): return ACTION
    p.infer = infer
    try:
        async with TestClient(TestServer(app)) as client:
            result = await post(client)
            result = await post(client,result)
            await asyncio.wait_for(builder.entered.wait(),2)
            for _ in range(6):
                result = await post(client,result)
            assert len(p.memory.pending) == 1
            job = p.memory.pending[SID]
            assert json.loads(job.state)["sequence"] == 8
            assert isinstance(job.state,bytes)
            original = job.state
            # A later state mutation cannot alter enqueued immutable work.
            p.sessions[SID].state["test"] = True
            assert job.state == original
            total = p.total()+p.memory.size
            p.c.global_bytes = total
            p.memory.submit(SID,p.sessions[SID])
            assert p.memory.pending[SID].state == original
            builder.release.set()
            await asyncio.wait_for(builder.finished.wait(),2)
    finally:
        builder.release.set()


@pytest.mark.asyncio
async def test_worker_failure_preserves_last_completed_snapshot():
    builder = SnapshotBuilder("thumbnail")
    app = create_app(Config(SECRET,"test",protocol="tiered-v1",memory_interval=0),builder)
    p = app["proxy"]
    async def infer(messages): return ACTION
    p.infer = infer
    try:
        async with TestClient(TestServer(app)) as client:
            first = await post(client)
            prior = b'{"through":1}'
            p.sessions[SID].memory_snapshot = prior
            async def fail(**kwargs):
                builder.finished.set()
                raise ValueError("secret output")
            builder.compact = fail
            await post(client,first)
            await asyncio.wait_for(builder.finished.wait(),2)
            assert p.sessions[SID].memory_snapshot == prior
            assert p.memory.failures == 1
    finally:
        builder.release.set()


@pytest.mark.asyncio
async def test_real_upstream_note_connection_does_not_occupy_action_capacity():
    from aiohttp import web
    entered, release = asyncio.Event(), asyncio.Event()
    payloads = []
    async def completion(request):
        payload = await request.json()
        payloads.append(payload)
        memory = payload["messages"][0]["content"] == "genuine note job"
        if memory:
            entered.set()
            await release.wait()
        return web.json_response({"choices":[{
            "finish_reason":"stop","message":{"role":"assistant",
                "content":"Observed uncertain destination." if memory else ACTION}}]})
    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions",completion)
    server = TestServer(upstream)
    await server.start_server()
    builder = SnapshotBuilder("note")
    app = create_app(Config(SECRET,"fixture",protocol="tiered-v1",memory_interval=0,
                           max_inflight=1,upstream=str(server.make_url("/v1/chat/completions"))),builder)
    try:
        async with TestClient(TestServer(app)) as client:
            result = await post(client)
            result = await post(client,result)
            await asyncio.wait_for(entered.wait(),2)
            await asyncio.wait_for(post(client,result),2)
            assert not release.is_set()
            assert len(payloads)==4
            assert sum("stop" in payload for payload in payloads)==3
            assert len([p for p in payloads if "stop" not in p])==1
            release.set()
            await asyncio.wait_for(builder.finished.wait(),2)
    finally:
        release.set()
        builder.release.set()
        await server.close()


@pytest.mark.asyncio
async def test_cpu_cancellation_drains_one_job_before_next_submission(monkeypatch):
    from SessionProxy import memory
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    def blocked():
        loop.call_soon_threadsafe(started.set)
        release.wait()
        return "completed CPU work"
    task = asyncio.create_task(memory._cpu_call(blocked))
    try:
        await asyncio.wait_for(started.wait(),2)
        task.cancel()
        # A loop callback proves cancellation was delivered, but the owned job
        # remains retained until its one running CPU call drains.
        delivered = asyncio.Event()
        loop.call_soon(delivered.set)
        await delivered.wait()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task,return_exceptions=True)


@pytest.mark.asyncio
async def test_background_admission_failure_does_not_fail_committed_action(monkeypatch):
    builder = SnapshotBuilder("note")
    app = create_app(Config(SECRET,"fixture",protocol="tiered-v1",memory_interval=0),builder)
    p = app["proxy"]
    async def infer(messages): return ACTION
    p.infer = infer
    def fail(*args): raise RuntimeError("worker admission failure")
    monkeypatch.setattr(p.memory,"submit",fail)
    async with TestClient(TestServer(app)) as client:
        result = await post(client)
        assert p.sessions[SID].pending["action_id"] == result["action_id"]
        assert p.memory.failures == 1


@pytest.mark.asyncio
async def test_completed_snapshot_swap_does_not_mutate_inflight_prompt():
    builder = SnapshotBuilder("note")
    app = create_app(Config(SECRET,"fixture",protocol="tiered-v1",memory_interval=0),builder)
    p = app["proxy"]
    action_entered, release_action = asyncio.Event(), asyncio.Event()
    async def infer(messages):
        if len(builder.calls) > 2:
            action_entered.set()
            await release_action.wait()
        return ACTION
    p.infer = infer
    # Complete summary only after the third request has captured its prompt.
    async def summarize(messages):
        await action_entered.wait()
        return "model generated memory"
    p.infer_memory = summarize
    try:
        async with TestClient(TestServer(app)) as client:
            first = await post(client)
            second = await post(client,first)
            old_prompt = copy.deepcopy(p.sessions[SID].messages)
            running = asyncio.create_task(post(client,second))
            await asyncio.wait_for(builder.finished.wait(),2)
            assert not running.done()
            snapshot = p.sessions[SID].memory_snapshot
            assert isinstance(snapshot,bytes)
            assert p.sessions[SID].messages == old_prompt
            assert builder.calls[-1] is None
            p.memory.closed = True  # isolate this publication from the next job
            release_action.set()
            await asyncio.wait_for(running,2)
            assert p.sessions[SID].memory_snapshot == snapshot
    finally:
        release_action.set()
        builder.release.set()
