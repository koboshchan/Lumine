"""Bounded, authenticated Lumine session protocol. No model weights are loaded."""
import asyncio
import base64
import copy
import hmac
import importlib
import io
import inspect
import ipaddress
import json
import os
import re
import secrets
import time
import warnings
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web
from PIL import Image

try:
    from .background import MemoryWorker
except ImportError:  # direct offline test import
    from background import MemoryWorker

HEX = re.compile(r"[0-9a-f]{32}\Z")
KEYS = set("Esc Tab Caps Shift Ctrl Alt Space Back Enter Left Up Right Down LB RB MB".split())
KEYS.update("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
KEYS.update("zero one two three four five six seven eight nine".split())
KEYS.update("One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve".split())
PREFIX = "data:image/png;base64,"
MAX_IMAGE = 4 * 1024 * 1024
MAX_BODY = 6 * 1024 * 1024
MAX_COMPLETION = 32 * 1024


def parse_completion(text, tiered=False):
    if not isinstance(text, str) or len(text.encode()) > MAX_COMPLETION:
        raise ValueError("completion size")
    s = text.strip()
    if s.endswith("<|im_end|>"):
        s = s[:-10].strip()
    note = None
    if tiered and "<|memory|>" in s:
        action, tail = s.split("<|memory|>", 1)
        if not tail.endswith("</|memory|>"):
            raise ValueError("memory delimiter")
        note = tail[:-11]
        if not note or len(note.encode()) > 8192 or "<|" in note or "|>" in note:
            raise ValueError("memory body")
        s = action.strip()
    if not s.startswith("<|action_start|>") or not s.endswith("<|action_end|>"):
        raise ValueError("action delimiter")
    fields = s[16:-14].split(";")
    if len(fields) != 7:
        raise ValueError("action fields")
    move = fields[0].split()
    absolute = bool(move and move[0] == "<|use_abs|>")
    if absolute:
        move = move[1:]
    if len(move) != 3 or any(not re.fullmatch(r"[+-]?[0-9]+", v) or len(v) > 20 for v in move):
        raise ValueError("movement")
    x, y, wheel = map(int, move)
    if not -5 <= wheel <= 5:
        raise ValueError("wheel")
    if absolute:
        if not (0 <= x <= 1279 and 0 <= y <= 719):
            raise ValueError("absolute")
    elif not (-995 <= x <= 995 and -996 <= y <= 996 and x % 5 == 0 and y % 4 == 0):
        raise ValueError("relative")
    for chunk in fields[1:]:
        keys = chunk.split()
        ks = set(keys)
        if len(keys) > 4 or len(ks) != len(keys) or not ks <= KEYS:
            raise ValueError("keys")
        if ("Alt" in ks and ks & {"Tab", "Four", "Esc"}) or {"Ctrl", "Esc"} <= ks:
            raise ValueError("OS shortcut")
    return s, note


def decode_frame(frame):
    if not isinstance(frame, str) or not frame.startswith(PREFIX) or len(frame) > MAX_BODY - 1024:
        raise ValueError("PNG data URL required")
    raw = base64.b64decode(frame[len(PREFIX):], validate=True)
    if len(raw) > MAX_IMAGE or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("PNG bytes")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != "PNG" or im.size not in {(1280, 720), (1280, 736)} or getattr(im, "n_frames", 1) != 1:
                raise ValueError("PNG dimensions")
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            if len(im.tobytes()) > 1280 * 736 * 4:
                raise ValueError("decoded pixels")
            rgb = im.convert("RGB")
            if rgb.height == 736 and rgb.crop((0, 720, 1280, 736)).getbbox() is not None:
                raise ValueError("nonblack padding")
            padded = Image.new("RGB", (1280, 736))
            padded.paste(rgb.crop((0, 0, 1280, 720)), (0, 0))
            out = io.BytesIO()
            padded.save(out, format="PNG")
    return PREFIX + base64.b64encode(out.getvalue()).decode()


def packed(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


@dataclass
class Config:
    secret: str
    model: str
    upstream: str = "http://127.0.0.1:8000/v1/chat/completions"
    protocol: str = "image-only"
    ttl: float = 900
    max_sessions: int = 8
    session_bytes: int = 24 * 1024 * 1024
    global_bytes: int = 96 * 1024 * 1024
    max_requests: int = 8192
    input_tokens: int = 30000
    timeout: float = 30
    max_inflight: int = 2
    upstream_key: str = ""
    memory_timeout: float = 30
    memory_interval: float = 2

    def validate(self):
        if not isinstance(self.secret, str) or not 32 <= len(self.secret.encode()) <= 512 or len(set(self.secret)) < 12 or not self.secret.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in self.secret):
            raise ValueError("strong LUMINE_PROXY_SECRET required (32+ bytes)")
        if not self.model or len(self.model) > 256 or self.protocol not in {"image-only", "tiered-v1"}:
            raise ValueError("model/protocol")
        u = urlsplit(self.upstream)
        try:
            local = ipaddress.ip_address(u.hostname).is_loopback
        except (ValueError, TypeError):
            local = False
        if not local or u.scheme != "http" or u.username or u.password or u.query or u.fragment or u.path != "/v1/chat/completions":
            raise ValueError("upstream must be literal loopback HTTP /v1/chat/completions")
        if any(v <= 0 for v in (self.ttl, self.max_sessions, self.session_bytes, self.global_bytes, self.max_requests, self.input_tokens, self.timeout, self.max_inflight, self.memory_timeout)):
            raise ValueError("positive limits required")
        if self.memory_interval < 0:
            raise ValueError("memory interval must be nonnegative")


@dataclass
class Session:
    touched: float
    busy: bool = False
    state: dict = field(default_factory=dict)
    pending: dict | None = None
    requests: set = field(default_factory=set)
    messages: list = field(default_factory=list)
    compacted: float = 0
    epoch: str = field(default_factory=lambda: secrets.token_hex(16))
    memory_snapshot: bytes | None = None
    memory_enqueued: float = float("-inf")

    def size(self):
        return len(packed([self.state, self.pending, sorted(self.requests), self.messages])) + len(self.memory_snapshot or b"")


class Proxy:
    def __init__(self, config, builder, clock=time.monotonic):
        config.validate()
        if builder is None:
            raise ValueError("shared exact-processor builder is required")
        self.c = config
        self.builder = builder
        self.clock = clock
        self.sessions = {}
        self.inflight = 0
        self.reserved_bytes = 0
        self.client = None
        self.memory_client = None
        self.memory = MemoryWorker(self)

    def expire(self):
        now = self.clock()
        for sid, s in list(self.sessions.items()):
            if not s.busy and now - s.touched >= self.c.ttl:
                del self.sessions[sid]
                self.memory.discard(sid)

    def total(self):
        return sum(s.size() for s in self.sessions.values())

    async def infer_memory(self, messages):
        return await self._complete(messages, memory=True)

    async def infer(self, messages):
        return await self._complete(messages, memory=False)

    async def _complete(self, messages, *, memory):
        headers = {"Authorization": "Bearer " + self.c.upstream_key} if self.c.upstream_key else {}
        payload = {
            "model": self.c.model, "messages": messages, "max_tokens": 512,
            "temperature": 0, "stream": False, "skip_special_tokens": False,
        }
        if not memory:
            payload["stop"] = ["<|memory|>"]  # note generation belongs to the worker only
        client = self.memory_client if memory else self.client
        async with client.post(self.c.upstream, headers=headers, json=payload, allow_redirects=False) as r:
            if r.status != 200:
                raise ValueError("upstream status")
            body = bytearray()
            async for chunk in r.content.iter_chunked(8192):
                body.extend(chunk)
                if len(body) > MAX_COMPLETION:
                    raise ValueError("upstream body size")
            value = json.loads(body)
            choices = value["choices"]
            if not isinstance(choices, list) or len(choices) != 1 or choices[0].get("finish_reason") != "stop":
                raise ValueError("incomplete choice")
            message = choices[0]["message"]
            if message.get("tool_calls") or message.get("function_call") or message.get("refusal") or message.get("role") != "assistant":
                raise ValueError("non-action response")
            return message["content"]

    async def step(self, request):
        data = await read_json(request, {"session_id", "request_id", "frame", "last_executed_action", "executed_action_id"})
        sid, rid = data["session_id"], data["request_id"]
        if not valid_id(sid) or not valid_id(rid):
            raise web.HTTPBadRequest()
        self.expire()
        s = self.sessions.get(sid)
        if s and s.busy:
            raise web.HTTPConflict()
        if s and rid in s.requests:
            raise web.HTTPConflict()
        if self.inflight >= self.c.max_inflight:
            raise web.HTTPServiceUnavailable()
        ack, aid = data["last_executed_action"], data["executed_action_id"]
        if (ack is None) != (aid is None):
            raise web.HTTPConflict()
        if ack is not None and (not s or not s.pending or ack != s.pending["action"] or aid != s.pending["action_id"]):
            raise web.HTTPConflict()
        try:
            frame = decode_frame(data["frame"])
        except Exception:
            raise web.HTTPBadRequest() from None
        if s is None:
            if len(self.sessions) >= self.c.max_sessions:
                raise web.HTTPServiceUnavailable()
            s = Session(self.clock(), compacted=self.clock())
            self.sessions[sid] = s
        if len(s.requests) >= self.c.max_requests:
            raise web.HTTPConflict(text="session request limit; start a new session")
        if self.total() + self.memory.size + self.reserved_bytes + 64 > self.c.global_bytes:
            raise web.HTTPServiceUnavailable()
        # Burn the id before inference. Never roll this anti-replay metadata back.
        s.requests.add(rid)
        s.busy = True
        self.inflight += 1
        reservation = 0
        try:
            # The builder receives only an acknowledged prediction, never unexecuted output.
            executed = copy.deepcopy(s.pending) if ack is not None else None
            kwargs = {}
            if getattr(self.builder, "snapshot_mode", False):
                kwargs["memory_snapshot"] = s.memory_snapshot
            plan = self.builder(
                state=copy.deepcopy(s.state), frame=frame, executed=executed,
                now=self.clock(), protocol=self.c.protocol, max_input_tokens=self.c.input_tokens,
                allow_compaction=False if kwargs else self.clock() - s.compacted >= 30, **kwargs,
            )
            if not isinstance(plan, dict) or not {"state", "messages", "input_tokens", "compacted"} <= set(plan) or set(plan) - {"state", "messages", "input_tokens", "compacted", "pre_compaction_tokens"}:
                raise ValueError("builder contract")
            messages = plan["messages"]
            if type(plan["input_tokens"]) is not int or not 0 < plan["input_tokens"] <= self.c.input_tokens:
                raise ValueError("exact processor budget")
            if not isinstance(messages, list) or not messages or not isinstance(plan["state"], dict) or type(plan["compacted"]) is not bool:
                raise ValueError("builder types")
            if plan["compacted"] and not kwargs and self.clock() - s.compacted < 30 and not (ack is None and s.pending is not None) and not (type(plan.get("pre_compaction_tokens")) is int and plan["pre_compaction_tokens"] > self.c.input_tokens):
                raise ValueError("early compaction")
            if self.c.protocol == "tiered-v1" and s.messages and not plan["compacted"] and messages[:len(s.messages)] != s.messages:
                raise ValueError("non-append prefix")
            candidate_bytes = len(packed([plan["state"], messages])) + MAX_COMPLETION + s.size()
            if candidate_bytes + self.memory.session_size(sid) > self.c.session_bytes or self.total() + self.reserved_bytes + self.memory.size + candidate_bytes > self.c.global_bytes:
                raise web.HTTPRequestEntityTooLarge(max_size=self.c.session_bytes, actual_size=candidate_bytes)
            reservation = candidate_bytes
            self.reserved_bytes += reservation
            async with asyncio.timeout(self.c.timeout):
                completion = await self.infer(messages)
            action, note = parse_completion(completion, self.c.protocol == "tiered-v1")
            if getattr(self.builder, "snapshot_mode", False):
                note = None  # never turn action generation into a memory-writing task
            pending = {"action": action, "action_id": secrets.token_hex(16), "note": note,
                       "note_provenance": "model-generated", "frame": frame}
            # Detect a dropped client before committing application state.
            if request.transport is None or request.transport.is_closing():
                raise asyncio.CancelledError()
            candidate = Session(self.clock(), state=plan["state"], pending=pending,
                                requests=s.requests, messages=messages, compacted=self.clock() if plan["compacted"] else s.compacted,
                                epoch=s.epoch, memory_snapshot=s.memory_snapshot, memory_enqueued=s.memory_enqueued)
            if candidate.size() + self.memory.session_size(sid) > self.c.session_bytes or self.total() - s.size() + candidate.size() + self.memory.size > self.c.global_bytes:
                raise web.HTTPServiceUnavailable()
            self.sessions[sid] = candidate
            try:
                self.memory.submit(sid, candidate)
            except Exception:
                # Admission is best-effort; a background scheduling failure must
                # not turn an already committed action into an ambiguous error.
                self.memory.failures += 1
            return web.json_response({"session_id": sid, "action": action,
                                     "action_id": pending["action_id"], "protocol": "lumine-session-v1"})
        except web.HTTPException:
            raise
        except (asyncio.CancelledError,):
            raise
        except Exception:
            raise web.HTTPBadGateway(text="inference or builder failed") from None
        finally:
            s.busy = False
            self.inflight -= 1
            self.reserved_bytes -= reservation

    async def reset(self, request):
        data = await read_json(request, {"session_id"})
        sid = data["session_id"]
        if not valid_id(sid):
            raise web.HTTPBadRequest()
        self.expire()
        if sid in self.sessions and self.sessions[sid].busy:
            raise web.HTTPConflict()
        # Keep bounded replay tombstones until TTL, preventing reset/retry replay.
        self.memory.discard(sid)
        if sid in self.sessions:
            s = self.sessions[sid]
            self.sessions[sid] = Session(self.clock(), requests=s.requests, compacted=self.clock())
        return web.json_response({"reset": True})


def valid_id(value):
    return isinstance(value, str) and HEX.fullmatch(value) is not None


async def read_json(request, keys):
    if request.content_type != "application/json" or request.headers.get("Content-Encoding", "identity") != "identity":
        raise web.HTTPBadRequest()
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError("duplicate")
            out[k] = v
        return out
    try:
        data = json.loads(await request.read(), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError()
        return data
    except web.HTTPException:
        raise
    except Exception:
        raise web.HTTPBadRequest() from None


def create_app(config, builder, clock=time.monotonic):
    proxy = Proxy(config, builder, clock)
    @web.middleware
    async def auth(request, handler):
        expected = ("Bearer " + config.secret).encode()
        supplied = request.headers.get("Authorization", "").encode()
        if len(supplied) > 1024 or not hmac.compare_digest(supplied, expected):
            raise web.HTTPUnauthorized()
        return await handler(request)
    app = web.Application(middlewares=[auth], client_max_size=MAX_BODY)
    app["proxy"] = proxy
    app.router.add_post("/v1/session/step", proxy.step)
    app.router.add_post("/v1/session/reset", proxy.reset)
    async def lifecycle(app):
        proxy.client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.timeout), trust_env=False,
            auto_decompress=False, connector=aiohttp.TCPConnector(limit=config.max_inflight))
        proxy.memory_client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.memory_timeout), trust_env=False,
            auto_decompress=False, connector=aiohttp.TCPConnector(limit=1))
        proxy.memory.start()
        async def sweeper():
            while True:
                await asyncio.sleep(min(config.ttl, 30))
                proxy.expire()
        task = asyncio.create_task(sweeper())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await proxy.memory.close()
            await proxy.memory_client.close()
            cleanup = getattr(proxy.builder, "close", None)
            if cleanup:
                result = cleanup()
                if inspect.isawaitable(result):
                    await result
            await proxy.client.close()
    app.cleanup_ctx.append(lifecycle)
    return app


def main():
    config = Config(secret=os.environ.get("LUMINE_PROXY_SECRET", ""),
                    model=os.environ.get("LUMINE_MODEL", ""),
                    upstream=os.environ.get("LUMINE_UPSTREAM", "http://127.0.0.1:8000/v1/chat/completions"),
                    protocol=os.environ.get("LUMINE_MODEL_PROTOCOL", "image-only"),
                    upstream_key=os.environ.get("LUMINE_UPSTREAM_KEY", ""),
                    memory_timeout=float(os.environ.get("LUMINE_MEMORY_TIMEOUT", "30")),
                    memory_interval=float(os.environ.get("LUMINE_MEMORY_INTERVAL", "2")))
    config.validate()
    # This must resolve to a shared Lumine-repo exact AutoProcessor implementation.
    module = os.environ.get("LUMINE_PROMPT_ADAPTER", "SessionProxy.adapter:build_step")
    if not module:
        raise ValueError("LUMINE_PROMPT_ADAPTER module:function required; shared builder unavailable")
    name, function = module.split(":")
    builder = getattr(importlib.import_module(name), function)
    web.run_app(create_app(config, builder), host="127.0.0.1",
                port=int(os.environ.get("LUMINE_PROXY_PORT", "8766")), handler_cancellation=True)


if __name__ == "__main__":
    main()
