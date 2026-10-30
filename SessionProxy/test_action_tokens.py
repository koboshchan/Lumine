"""Actual 4B protocol IDs; exact IDs, not a vocabulary count."""
import json
import os
from pathlib import Path

import pytest
from SessionProxy.adapter import validate_custom_token_ids
from SessionProxy.service import KEYS, parse_completion

EXPECTED = json.loads(Path(__file__).with_name("action_vocab_ids.json").read_text())
CUSTOM = json.loads(Path(__file__).with_name("custom_token_ids.json").read_text())
CHECKPOINT_HASHES = {
    "action-token-ids.json": "d034ec035aad9c0fe65bb4287c4c22fb9b51ac8034ca2bd4397ec05ffa31a2d9",
    "tokenizer.json": "58e7028ff56a78adc38d7989adc6b2a00bed36a257b9623c0fed76c8e8e6f030",
    "tokenizer_config.json": "15d9c5ecb9cfd3c63e5be36629d00b7e21f32ce7c54037fc7fdb02033bca1d13",
    "chat_template.jinja": "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4",
}


def action(key="", absolute=False):
    movement = "<|use_abs|> 1279 719 -5" if absolute else "-995 996 5"
    return "<|action_start|>" + movement + " ; " + " ; ".join([key] * 6) + "<|action_end|>"


def test_atomic_but_wrong_ids_fail_closed():
    class Tokenizer:
        def __init__(self, shift):
            self.vocab = {k: v + shift for k, v in EXPECTED.items()}
        def encode(self, token, **kw):
            return [self.vocab[token]]
        def convert_tokens_to_ids(self, token):
            return self.vocab[token]
        def decode(self, ids, **kw):
            return next(k for k, v in self.vocab.items() if v == ids[0])
    assert len(CUSTOM) == 7
    assert set(CUSTOM) == {"<|action_start|>", "<|action_end|>", "<|use_abs|>",
                           "<|thought_start|>", "<|thought_end|>", "Eleven", "Twelve"}
    validate_custom_token_ids(Tokenizer(0))
    base_only_changed = Tokenizer(0)
    for key in EXPECTED.keys() - CUSTOM.keys():
        base_only_changed.vocab[key] += 1
    validate_custom_token_ids(base_only_changed)  # thirty words aren't custom checks
    with pytest.raises(ValueError, match="IDs differ"):
        validate_custom_token_ids(Tokenizer(1))


def test_parser_all_action_vocabulary_entries_and_no_thought_execution():
    assert len(EXPECTED) == 37 and len(set(EXPECTED.values())) == 37
    structural = {"<|action_start|>", "<|action_end|>", "<|use_abs|>",
                  "<|thought_start|>", "<|thought_end|>"}
    for key in EXPECTED.keys() - structural:
        assert key in KEYS
        for absolute in (False, True):
            text = action(key, absolute)
            assert parse_completion(text) == (text, None)
            assert parse_completion(text + "<|im_end|>") == (text, None)
    for token in ("<|thought_start|>", "<|thought_end|>"):
        with pytest.raises(ValueError):
            parse_completion(token + action())
        with pytest.raises(ValueError):
            parse_completion(action(token))


@pytest.mark.skipif(not os.environ.get("LUMINE_TEST_PROCESSOR"),
                    reason="saved test checkpoint tokenizer required for exact-ID test")
def test_real_checkpoint_ids_rendered_ack_and_decoded_output(monkeypatch):
    import hashlib
    from transformers import AutoTokenizer
    from SessionProxy.adapter import _tokenizer, build_step
    from lumine_memory.live import _native
    from lumine_memory.prompt import render_text

    root = Path(os.environ["LUMINE_TEST_PROCESSOR"])
    # Verify this is the hash-pinned current 4B marker artifact.
    for name, expected_hash in CHECKPOINT_HASHES.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected_hash
    assert json.loads((root / "action-token-ids.json").read_text()) == EXPECTED
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    validate_custom_token_ids(tokenizer)
    monkeypatch.setenv("LUMINE_PROCESSOR_DIR", str(root))
    _tokenizer.cache_clear()
    try:
        active = _tokenizer()
        validate_custom_token_ids(active)
        structural = {"<|action_start|>", "<|action_end|>", "<|use_abs|>",
                      "<|thought_start|>", "<|thought_end|>"}
        contextual_keys = EXPECTED.keys() - structural
        merged = {key for key in contextual_keys
                  if EXPECTED[key] not in active.encode(action(key), add_special_tokens=False)}
        assert merged == contextual_keys - {"Eleven", "Twelve"}
        for key in contextual_keys:
            for absolute in (False, True):
                text = action(key, absolute)
                encoded = active.encode(text, add_special_tokens=False)
                assert active.decode(encoded, skip_special_tokens=False) == text
                # Thirty pre-existing vocabulary words merge with leading
                # whitespace. Preserve checkpoint/training serialization rather
                # than inventing 37 newly atomic in-context tokens.
                assert encoded == tokenizer.encode(text, add_special_tokens=False)
                if key in {"Eleven", "Twelve"}:
                    assert EXPECTED[key] in encoded
                assert encoded[0] == EXPECTED["<|action_start|>"]
                assert encoded[-1] == EXPECTED["<|action_end|>"]
                if absolute:
                    assert encoded[1] == EXPECTED["<|use_abs|>"]
                assert parse_completion(active.decode(encoded, skip_special_tokens=False)) == (text, None)
                first = build_step(state={}, frame="frame1", executed=None, now=1,
                                   protocol="tiered-v1", max_input_tokens=30000,
                                   allow_compaction=False)
                assert not any(m["role"] == "assistant" for m in first["messages"])
                ack = {"frame": "frame1", "action": text, "action_id": "a" * 32}
                second = build_step(state=first["state"], frame="frame2", executed=ack, now=1.2,
                                    protocol="tiered-v1", max_input_tokens=30000,
                                    allow_compaction=False)
                assert [m["content"] for m in second["messages"] if m["role"] == "assistant"] == [text]
                rendered = render_text(_native(second["messages"]), active.apply_chat_template, [920, 920])
                ids = active.encode(rendered, add_special_tokens=False)
                start = ids.index(EXPECTED["<|action_start|>"])
                assert ids[start:start + len(encoded)] == encoded
                third = build_step(state=second["state"], frame="frame3", executed=None, now=1.4,
                                   protocol="tiered-v1", max_input_tokens=30000,
                                   allow_compaction=False)
                assert [m["content"] for m in third["messages"] if m["role"] == "assistant"] == [text]
    finally:
        _tokenizer.cache_clear()


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("LUMINE_TEST_PROCESSOR"),
                    reason="saved checkpoint tokenizer required")
async def test_real_http_only_matching_executed_ack_enters_history(monkeypatch):
    import base64
    import io
    import secrets
    from aiohttp.test_utils import TestClient, TestServer
    from PIL import Image
    from SessionProxy.adapter import _tokenizer, build_step
    from SessionProxy.service import Config, create_app

    monkeypatch.setenv("LUMINE_PROCESSOR_DIR", os.environ["LUMINE_TEST_PROCESSOR"])
    _tokenizer.cache_clear()
    out = io.BytesIO()
    Image.new("RGB", (1280, 720)).save(out, format="PNG")
    frame = "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()
    key = secrets.token_urlsafe(40)
    app = create_app(Config(key, "test", protocol="tiered-v1"), build_step)
    proxy = app["proxy"]
    text = action("LB", True)

    async def infer(messages):
        return text
    proxy.infer = infer

    try:
        async with TestClient(TestServer(app)) as client:
            sid = secrets.token_hex(16)
            auth = {"Authorization": "Bearer " + key}

            async def step(ack=None, action_id=None):
                response = await client.post("/v1/session/step", headers=auth, json={
                    "session_id": sid, "request_id": secrets.token_hex(16), "frame": frame,
                    "last_executed_action": ack, "executed_action_id": action_id})
                return response, await response.json() if response.status == 200 else None

            response, first = await step()
            assert response.status == 200
            assert not any(m["role"] == "assistant" for m in proxy.sessions[sid].state["messages"])
            response, second = await step()  # first prediction never executed
            assert response.status == 200
            assert not any(m["role"] == "assistant" for m in proxy.sessions[sid].state["messages"])
            assert (await step(text, first["action_id"]))[0].status == 409
            response, third = await step(text, second["action_id"])
            assert response.status == 200
            history = [m["content"] for m in proxy.sessions[sid].state["messages"]
                       if m["role"] == "assistant"]
            assert history == [text]
            assert proxy.sessions[sid].pending["action_id"] == third["action_id"]
    finally:
        _tokenizer.cache_clear()


@pytest.mark.skipif(not os.environ.get('LUMINE_TEST_PROCESSOR'), reason='hash-checked saved checkpoint required')
def test_before_after_seven_custom_change_preserves_all_30_contextual_words(monkeypatch):
    import hashlib
    from transformers import AutoTokenizer
    from SessionProxy.adapter import _tokenizer, build_step
    from lumine_memory.live import _native
    from lumine_memory.prompt import render_text
    root=Path(os.environ['LUMINE_TEST_PROCESSOR'])
    for n,h in CHECKPOINT_HASHES.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
    monkeypatch.setenv('LUMINE_PROCESSOR_DIR', str(root))
    _tokenizer.cache_clear()
    t=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    active=_tokenizer()
    keys=sorted(set(EXPECTED)-{'<|action_start|>','<|action_end|>','<|use_abs|>','<|thought_start|>','<|thought_end|>','Eleven','Twelve'})
    rows=[]
    for key in keys:
     for absolute in (False,True):
      for position in range(6):
       chunks=['']*6;chunks[position]=key
       movement='<|use_abs|> 1279 719 -5' if absolute else '-995 996 5'
       text='<|action_start|>'+movement+' ; '+' ; '.join(chunks)+'<|action_end|>'
       first=build_step(state={},frame='f1',executed=None,now=1,protocol='tiered-v1',max_input_tokens=30000,allow_compaction=False)
       second=build_step(state=first['state'],frame='f2',executed={'frame':'f1','action':text},now=1.2,protocol='tiered-v1',max_input_tokens=30000,allow_compaction=False)
       rendered=render_text(_native(second['messages']),active.apply_chat_template,[920,920])
       ids=active.encode(text,add_special_tokens=False);prompt_ids=active.encode(rendered,add_special_tokens=False)
       assert ids==t.encode(text,add_special_tokens=False)
       assert active.decode(ids,skip_special_tokens=False)==text
       rows.append({'key':key,'absolute':absolute,'position':position,'text':text,'ids':ids,'prompt_ids':prompt_ids})
    new={k:v for k,v in EXPECTED.items() if v>=151669}
    assert len(new)==7
    for k,v in new.items():assert active.encode(k,add_special_tokens=False)==[v]

    raw=json.dumps({'custom_ids':new,'rows':rows},sort_keys=True,separators=(',',':')).encode()
    # All 360 contextual action encodings must round trip without rewriting BPE.
    assert len(rows)==360
    assert all(active.decode(r['ids'], skip_special_tokens=False) == r['text'] for r in rows)
    _tokenizer.cache_clear()

@pytest.mark.parametrize("marker", ["<use_absolute>", "<use_abs>"])
def test_legacy_absolute_marker_rejected(marker):
    with pytest.raises(ValueError):
        parse_completion(action(absolute=True).replace("<|use_abs|>", marker))
