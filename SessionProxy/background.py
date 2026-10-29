"""One bounded, coalescing memory worker. Only completed immutable bytes publish."""
import asyncio
import json
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    sid: str
    epoch: str
    state: bytes
    boundary: float


class MemoryWorker:
    def __init__(self, proxy):
        self.proxy = proxy
        self.pending = OrderedDict()
        self.active = None
        self.wake = asyncio.Event()
        self.task = None
        self.closed = False
        self.failures = 0

    @property
    def size(self):
        return sum(len(j.state) for j in self.pending.values()) + (
            len(self.active.state) if self.active else 0)

    def discard(self, sid):
        self.pending.pop(sid, None)
        # In-flight model/CPU work never owns a session or a publication right.
        # Its epoch/expiry is rechecked after completion.

    def session_size(self, sid):
        return sum(len(j.state) for j in self.pending.values() if j.sid == sid) + (
            len(self.active.state) if self.active and self.active.sid == sid else 0)

    def submit(self, sid, session):
        p = self.proxy
        if self.closed or p.c.protocol != "tiered-v1" or not getattr(p.builder, "snapshot_mode", False):
            return
        if not session.state or not any(r.get("action") for r in session.state.get("records", [])):
            return
        now = p.clock()
        if sid not in self.pending and now - session.memory_enqueued < p.c.memory_interval:
            return
        raw = json.dumps(session.state, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode()
        previous = self.pending.get(sid)
        extra = len(raw) - (len(previous.state) if previous else 0)
        # Queue and active copies count toward both per-session and global limits.
        active_bytes = len(self.active.state) if self.active and self.active.sid == sid else 0
        if (session.size() + active_bytes + len(raw) > p.c.session_bytes or
                p.total() + p.reserved_bytes + self.size + extra > p.c.global_bytes):
            return  # Never wait for compaction or reclaim memory inline.
        self.pending[sid] = Job(sid, session.epoch, raw, now)
        session.memory_enqueued = now
        self.wake.set()

    def start(self):
        self.task = asyncio.create_task(self.run())

    async def close(self):
        self.closed = True
        self.pending.clear()
        self.wake.set()
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def run(self):
        while True:
            await self.wake.wait()
            self.wake.clear()
            while self.pending:
                _, job = self.pending.popitem(last=False)
                self.active = job
                p = self.proxy
                try:
                    s = p.sessions.get(job.sid)
                    if s is None or s.epoch != job.epoch or p.clock() - s.touched >= p.c.ttl:
                        continue
                    # Hooks must move image/token CPU work to their dedicated executor.
                    # No action/session lock is held over CPU or model work.
                    async with asyncio.timeout(p.c.memory_timeout):
                        result = await p.builder.compact(
                            state=json.loads(job.state), now=job.boundary,
                            summarize=p.infer_memory)
                    if result is None:
                        continue
                    raw = json.dumps(result, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()
                    if not isinstance(result, dict) or not result.get("through"):
                        raise ValueError("memory snapshot contract")
                    # Single event-loop publication, no await between check and swap.
                    s = p.sessions.get(job.sid)
                    if s is None or s.epoch != job.epoch or p.clock() - s.touched >= p.c.ttl:
                        continue
                    old_size = len(s.memory_snapshot or b"")
                    if (len(raw) + s.size() - old_size + self.session_size(job.sid) > p.c.session_bytes or
                            p.total() + p.reserved_bytes + self.size + len(raw) - old_size > p.c.global_bytes):
                        continue
                    old = json.loads(s.memory_snapshot) if s.memory_snapshot else {}
                    if int(result["through"]) > int(old.get("through", 0)):
                        s.memory_snapshot = raw  # immutable completed snapshot
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self.failures += 1  # never expose raw model output or replace good memory
                finally:
                    self.active = None
