"""Pure contract tests. Optional real-tokenizer mode is enabled via environment."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from lumine_memory.live import build_live_step, _native
from lumine_memory.prompt import render_text

ACTION = "<|action_start|>0 0 0;;;;;;<|action_end|>"


def native(messages, tokenize=False, add_generation_prompt=True):
    text = ""
    for m in messages:
        text += "<|im_start|>" + m["role"] + "\n"
        for c in (m["content"] if isinstance(m["content"], list) else [{"type":"text","text":m["content"]}]):
            text += "<|vision_start|><|image_pad|><|vision_end|>" if c["type"] == "image" else c["text"]
        text += "<|im_end|>\n"
    return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def fake_encode(s):
    # Fixture only. Real tokenizer suite below verifies exact budget independently.
    return [0] * (s.count("<|image_pad|>") + len(s.replace("<|image_pad|>","")) // 4)


def step(state=None, executed=None, now=1., template=native, encode=fake_encode, protocol="tiered-v1"):
    return build_live_step(state=state or {}, frame="full"+str(now), thumbnail="thumb"+str(now),
                           executed=executed, now=now, protocol=protocol,
                           max_input_tokens=30000, allow_compaction=False,
                           apply_template=template, encode=encode)


def ack(plan, note=None):
    return {"action":ACTION, "frame":plan["state"]["records"][-1]["image"],
            "note":note, "action_id":"a"*32}


def test_dryrun_observations_and_ack_prefix():
    first = step()
    second = step(first["state"], ack(first), 1.2)
    assert second["messages"][:len(first["messages"])] == first["messages"]
    assert not second["compacted"]
    third = step(second["state"], None, 1.4)
    assert third["compacted"] # no fabricated assistant for an unexecuted prediction
    assert sum(m["role"] == "assistant" for m in third["messages"]) == 1


def test_bounds_compaction_and_notes_provenance():
    plan=step()
    compacted=0
    for i in range(1,400):
        plan=step(plan["state"],ack(plan,"uncertain goal" if i == 2 else None),1+i*.2)
        compacted+=plan["compacted"]
        assert plan["input_tokens"]<=30000
        rows=plan["state"]["records"]
        assert sum(bool(r["image"]) for r in rows)<=16
        assert len(rows)<=170
    assert compacted>0
    assert plan["state"]["notes"][0]["provenance"] == "model-generated-unreviewed"
    assert not any("reviewed" in n for n in plan["state"]["notes"])


def test_note_budget_fail_closed_and_input_immutable():
    plan=step()
    before=json.dumps(plan)
    with pytest.raises(ValueError,match="genuine model compaction"):
        step(plan["state"],ack(plan,"x"*20000),1.2)
    assert json.dumps(plan)==before


def test_image_only_preserves_old_protocol():
    p=step(protocol="image-only")
    assert p["state"]=={} and len(p["messages"])==1
    assert len(p["messages"][0]["content"])==1
    assert p["messages"][0]["content"][0]["type"]=="image_url"


@pytest.mark.skipif(not os.environ.get("LUMINE_TEST_PROCESSOR"),reason="local saved tokenizer not supplied")
def test_real_checkpoint_tokenizer_budget_prefix_and_mixed_grids():
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(os.environ["LUMINE_TEST_PROCESSOR"],local_files_only=True,trust_remote_code=False)
    enc=lambda s:t.encode(s,add_special_tokens=False)
    plan=step(template=t.apply_chat_template,encode=enc)
    assert plan["input_tokens"]==len(enc(render_text(_native(plan["messages"]),t.apply_chat_template,[920])))
    old=enc(render_text(_native(plan["messages"]),t.apply_chat_template,plan["state"]["counts"]))
    nextplan=step(plan["state"],ack(plan),1.2,template=t.apply_chat_template,encode=enc)
    new=enc(render_text(_native(nextplan["messages"]),t.apply_chat_template,nextplan["state"]["counts"]))
    assert new[:len(old)]==old
    for i in range(2,90):
        plan=step(nextplan["state"],ack(nextplan),1+i*.2,template=t.apply_chat_template,encode=enc)
        nextplan=plan
        assert plan["input_tokens"]==len(enc(render_text(_native(plan["messages"]),t.apply_chat_template,plan["state"]["counts"])))
    assert 60 in plan["state"]["counts"] and 920 in plan["state"]["counts"]
    assert plan["state"]["generation"]>0


@pytest.mark.skipif(not os.environ.get("LUMINE_TEST_PROCESSOR"),reason="local saved processor not supplied")
def test_real_processor_full_and_thumbnail_grids():
    from transformers import AutoProcessor
    from PIL import Image
    p=AutoProcessor.from_pretrained(os.environ["LUMINE_TEST_PROCESSOR"],local_files_only=True,trust_remote_code=False)
    for size,tokens in [((1280,736),920),((320,192),60)]:
        v=p.image_processor(images=[Image.new("RGB",size)],do_resize=False,return_tensors="pt")
        assert int(v["image_grid_thw"].prod())//4==tokens


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("LUMINE_TEST_PROCESSOR"),reason="local saved tokenizer not supplied")
async def test_real_adapter_http_step_and_ack():
    import base64,io,secrets
    from aiohttp.test_utils import TestClient,TestServer
    from PIL import Image
    from SessionProxy.adapter import build_step, _tokenizer
    from SessionProxy.service import Config,create_app
    os.environ["LUMINE_PROCESSOR_DIR"]=os.environ["LUMINE_TEST_PROCESSOR"]
    _tokenizer.cache_clear()
    out=io.BytesIO()
    Image.new("RGB",(1280,720)).save(out,format="PNG")
    frame="data:image/png;base64,"+base64.b64encode(out.getvalue()).decode()
    key=secrets.token_urlsafe(40)
    for mode in ("image-only","tiered-v1"):
        app=create_app(Config(key,"test",protocol=mode),build_step)
        proxy=app["proxy"]
        async def infer(messages):
            return ACTION+("<|memory|>standing near an unknown place</|memory|>" if mode=="tiered-v1" else "")
        proxy.infer=infer # synthetic completion, real adapter+pixel preprocessing
        async with TestClient(TestServer(app)) as client:
            body={"session_id":secrets.token_hex(16),"request_id":secrets.token_hex(16),"frame":frame,
                  "last_executed_action":None,"executed_action_id":None}
            auth={"Authorization":"Bearer "+key}
            r=await client.post("/v1/session/step",json=body,headers=auth)
            assert r.status==200,await r.text()
            first=await r.json()
            body.update(request_id=secrets.token_hex(16),last_executed_action=first["action"],executed_action_id=first["action_id"])
            r=await client.post("/v1/session/step",json=body,headers=auth)
            assert r.status==200,await r.text()
            state=proxy.sessions[body["session_id"]].state
            if mode=="tiered-v1":
                assert len(state["records"])==2 and len(state["notes"])==1
                assert state["notes"][0]["provenance"]=="model-generated-unreviewed"
                assert state["notes"][0]["timestamp_ms"]<state["records"][-1]["timestamp_ms"]
