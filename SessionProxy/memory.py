"""Immutable boundary snapshots. Action assembly never decodes pixels or calls a model.

Service owns job admission/serialization. CPU phases use a dedicated single worker;
no background work ever mutates committed session state.
"""
import asyncio
import base64
import copy
import io
import json
import atexit
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from lumine_memory.live import _native
from lumine_memory.prompt import render_text

_CPU = None


def _executor():
    global _CPU
    if _CPU is None:
        _CPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lumine-memory")
    return _CPU


def shutdown():
    """Process-final cleanup hook; service must first cancel/join its jobs.

    A running CPU call may finish; do not block the event loop waiting for it."""
    global _CPU
    if _CPU is not None:
        _CPU.shutdown(wait=False, cancel_futures=True)
        _CPU = None


atexit.register(shutdown)


def _encode(t, text):
    return t.encode(text, add_special_tokens=False)


def count(t, messages, counts, *, generation=True):
    return len(_encode(t, render_text(_native(messages), t.apply_chat_template,
                                      counts, generation_prompt=generation)))


def observation(row, tier="current"):
    return {"role": "user", "content": [
        {"type": "text", "text": f'[{tier} {row["timestamp_ms"]}ms]'},
        {"type": "image_url", "image_url": {"url": row["thumbnail"] if tier == "middle" else row["image"]}}]}


def reply(row):
    text = row["action"]
    if row.get("note"):
        text += "\n<|memory|>" + row["note"] + "</|memory|>"
    return {"role": "assistant", "content": text}


def action_step(*, state, frame, executed, now, protocol, max_input_tokens,
                tokenizer, memory_snapshot=None):
    if protocol == "image-only":
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame}}]}]
        tokens = count(tokenizer, messages, [920])
        if tokens > max_input_tokens:
            raise ValueError("image-only token budget")
        return dict(state={}, messages=messages, input_tokens=tokens, compacted=False)
    if protocol != "tiered-v1":
        raise ValueError("unsupported model protocol")
    new = copy.deepcopy(state)
    records = new.setdefault("records", [])
    messages = new.setdefault("messages", [])
    counts = new.setdefault("counts", [])
    ts = max(int(now * 1000), new.get("last_ms", -1) + 1)
    if executed is not None:
        if not records or records[-1]["image"] != executed["frame"] or records[-1].get("action"):
            raise ValueError("execution must acknowledge latest observation exactly once")
        if not isinstance(executed.get("action"), str) or not executed["action"]:
            raise ValueError("empty executed action")
        records[-1].update(action=executed["action"], note=executed.get("note"))
        messages.append(reply(records[-1]))
    compacted = bool(messages and executed is None)
    if memory_snapshot is not None:
        if not isinstance(memory_snapshot, bytes):
            raise ValueError("immutable snapshot bytes required")
        snap = json.loads(memory_snapshot)
        through = int(snap["through"])
        if through > new.get("snapshot_through", 0):
            if through > new.get("sequence", 0) or snap["boundary_ms"] > ts:
                raise ValueError("future snapshot")
            # Only a completed ACK prefix can be replaced. Pending and all newer
            # observations survive, including discarded dry-run predictions.
            tail = [r for r in records if int(r["id"]) > through]
            messages = copy.deepcopy(snap["messages"])
            counts = list(snap["counts"])
            for row in tail:
                messages.append(observation(row))
                counts.append(920)
                if row.get("action"):
                    messages.append(reply(row))
            records = copy.deepcopy(snap["records"]) + tail
            new.update(snapshot_through=through, notes=copy.deepcopy(snap["notes"]))
            compacted = True
    seq = new.get("sequence", 0) + 1
    current = dict(id=str(seq), timestamp_ms=ts, image=frame, thumbnail="", action="")
    records.append(current)
    messages.append(observation(current))
    counts.append(920)
    tokens = count(tokenizer, messages, counts)
    if tokens > max_input_tokens:
        raise ValueError("hard token cap; background snapshot not ready")
    new.update(records=records, messages=messages, counts=counts, sequence=seq,
               last_ms=ts, max_input_tokens=max_input_tokens,
               generation=new.get("generation", 0) + int(compacted))
    return dict(state=new, messages=messages, input_tokens=tokens, compacted=compacted)


def thumbnail(url):
    # This function is called exclusively from the CPU preparation phase.
    from PIL import Image
    raw = base64.b64decode(url.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(raw)) as im:
        small = im.convert("RGB").crop((0, 0, 1280, 720)).resize((320, 180), Image.Resampling.LANCZOS)
        padded = Image.new("RGB", (320, 192), "black")
        padded.paste(small, (0, 0))
        out = io.BytesIO()
        padded.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def _notes_message(notes):
    return {"role": "user", "content": [{"type": "text", "text":
        "<|memory|>\n" + "\n".join(f'[{n["timestamp_ms"]}ms] {n["text"]}' for n in notes) + "\n</|memory|>"}]}


def _render(rows, notes):
    messages = [_notes_message(notes)] if notes else []
    counts = []
    for row in rows:
        tier = "middle" if not row["image"] else "recent"
        messages.extend((observation(row, tier), reply(row)))
        counts.append(60 if tier == "middle" else 920)
    return messages, counts


def prepare(state, now, tokenizer):
    boundary = int(now * 1000)
    eligible = [copy.deepcopy(r) for r in state.get("records", [])
                if r.get("action") and r["timestamp_ms"] < boundary]
    prior = state.get("snapshot_through", 0)
    fresh = [r for r in eligible if int(r["id"]) > prior]
    if not fresh:
        return None
    through = max(int(r["id"]) for r in fresh)
    recent = eligible[-16:]
    recent_ids = {r["id"] for r in recent}
    middle, last = [], None
    for row in eligible:
        if row["id"] in recent_ids or row["timestamp_ms"] < boundary - 300000:
            continue
        if last is None or row["timestamp_ms"] - last >= 2000:
            row["thumbnail"] = row.get("thumbnail") or thumbnail(row["image"])
            row["image"] = ""
            middle.append(row)
            last = row["timestamp_ms"]
    notes = copy.deepcopy(state.get("notes", []))
    budget = min(state.get("max_input_tokens", 30000), 30000)
    # Reserve space for the next observation and summary output, not merely this
    # ACK-only prefix. Remove oldest sparse evidence first, never recent16.
    rows = middle + recent
    while True:
        messages, counts = _render(rows, notes)
        instruction = {"role": "user", "content": [{"type": "text", "text":
            "Write a concise factual memory (at most 512 tokens). Merge the previous "
            "unreviewed memory with new observed evidence and actual executed actions. "
            "Keep uncertainty; do not invent events, goals, or reviewed labels. "
            f"Only evidence strictly before {boundary}ms is present."}]}
        evidence = messages + [instruction]
        if count(tokenizer, evidence, counts) <= budget - 1536:
            break
        if not middle:
            raise ValueError("memory evidence exceeds native token budget")
        middle.pop(0)
        rows = middle + recent
    return dict(through=through, boundary_ms=boundary, records=rows,
                previous_notes=notes, evidence=evidence, counts=counts, budget=budget)


def finish(prepared, text, tokenizer):
    if not isinstance(text, str) or not text.strip() or "<|" in text or "|>" in text:
        raise ValueError("genuine plain model summary required")
    if len(_encode(tokenizer, text)) > 512:
        raise ValueError("summary output exceeds 512 tokens")
    notes = [dict(timestamp_ms=prepared["boundary_ms"] - 1, text=text,
                  provenance="model-generated-unreviewed",
                  evidence_frame_ids=[r["id"] for r in prepared["records"]],
                  through=prepared["through"])]
    if len(_encode(tokenizer, _notes_message(notes)["content"][0]["text"])) > 2048:
        raise ValueError("memory notes exceed 2048 tokens")
    messages, counts = _render(prepared["records"], notes)
    tokens = count(tokenizer, messages, counts)
    if tokens + 1024 > prepared["budget"]:
        raise ValueError("snapshot exceeds native budget")
    return dict(through=prepared["through"], boundary_ms=prepared["boundary_ms"],
                records=prepared["records"], notes=notes, messages=messages,
                counts=counts, input_tokens=tokens)


async def _cpu_call(fn, *args):
    # asyncio timeout cannot stop a running native thread. Keep the one worker
    # occupied until that exact job drains, rather than stacking canceled CPU
    # submissions in ThreadPoolExecutor's otherwise unbounded queue.
    future = asyncio.get_running_loop().run_in_executor(_executor(), partial(fn, *args))
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(future)
        except Exception:
            pass
        raise


async def compact_state(*, state, now, summarize, tokenizer):
    prepared = await _cpu_call(prepare, state, now, tokenizer)
    if prepared is None:
        return None
    text = await summarize(prepared["evidence"])
    return await _cpu_call(finish, prepared, text, tokenizer)
