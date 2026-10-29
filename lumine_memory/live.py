"""Pure online tiered segments, shared template and tier selection with training.

Online notes are unreviewed model output, never relabelled as reviewed training data.
No summarizer is fabricated: excess notes fail closed until a genuine summary protocol
is available. This implementation does not claim learned hour-long retention.
"""
import copy
from types import SimpleNamespace
from .prompt import Frame, Policy, build_prompt, render_text, _messages


def _wire(messages):
    out = []
    for message in messages:
        if message["role"] == "assistant" and not message["content"]:
            continue  # Observation-only steps, especially dry run, have no action.
        m = copy.deepcopy(message)
        if isinstance(m["content"], list):
            for item in m["content"]:
                if item["type"] == "image":
                    item["type"] = "image_url"
                    item["image_url"] = {"url": item.pop("image")}
        out.append(m)
    return out


def _native(messages):
    out = copy.deepcopy(messages)
    for m in out:
        if isinstance(m["content"], list):
            for item in m["content"]:
                if item["type"] == "image_url":
                    item["type"] = "image"
                    item["image"] = item.pop("image_url")["url"]
    return out


def build_live_step(*, state, frame, executed, now, protocol, max_input_tokens,
                    allow_compaction, apply_template, encode, thumbnail=None):
    """Atomic proposal only. Caller commits it after successful parsed inference.

    thumbnail must be supplied for tiered mode, generated from validated pixels.
    State is JSON-safe, bounded by caller bytes/session and this retention policy.
    """
    if protocol not in ("image-only", "tiered-v1"):
        raise ValueError("unsupported model protocol")
    policy = Policy(input_tokens=max_input_tokens)
    def count(messages, counts):
        return len(encode(render_text(_native(messages), apply_template, counts)))
    current_id = str(state.get("sequence", 0) + 1)
    ts = max(int(now * 1000), state.get("last_ms", -1) + 1)
    current = Frame(current_id, "live", "train", ts, frame, thumbnail or "", "")
    if protocol == "image-only":
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame}}]}]
        tokens = count(messages, [920])
        if tokens > max_input_tokens:
            raise ValueError("image-only token budget")
        return {"state": {}, "messages": messages, "input_tokens": tokens, "compacted": False}
    if not thumbnail:
        raise ValueError("validated tiered thumbnail required")
    new = copy.deepcopy(state)
    records = new.get("records", [])
    notes = new.get("notes", [])
    messages = new.get("messages", [])
    counts = new.get("counts", [])
    if executed:
        if not records or executed["frame"] != records[-1]["image"]:
            raise ValueError("execution must acknowledge latest observation")
        records[-1]["action"] = executed["action"]
        assistant = executed["action"]
        if executed.get("note"):
            note = executed["note"]
            notes.append({"timestamp_ms": records[-1]["timestamp_ms"], "text": note,
                          "provenance": "model-generated-unreviewed",
                          "evidence_frame_ids": [records[-1]["id"]]})
            assistant += "\n<|memory|>" + note + "</|memory|>"
        messages.append({"role": "assistant", "content": assistant})
    notes = [n for n in notes if ts - n["timestamp_ms"] <= 3600000]
    if len(encode("\n".join(f'[{n["timestamp_ms"]}ms] {n["text"]}' for n in notes))) > policy.note_tokens:
        raise ValueError("notes require genuine model compaction, no synthetic summary")
    # Append current observation without changing any earlier segment message.
    current_messages, _, _ = _messages(current, [], [], [])
    candidate = messages + _wire(current_messages)
    candidate_counts = counts + [920]
    tokens = count(candidate, candidate_counts)
    pre_compaction_tokens = tokens
    # A discarded prediction has no real assistant reply to extend the native
    # generation prefix. Rebuild explicitly instead of inventing an action.
    compacted = bool(messages and (executed is None or allow_compaction or tokens > max_input_tokens))
    if compacted:
        inventory = [Frame(**r) for r in records] + [current]
        selected = build_prompt(inventory, current, [], apply_template=apply_template,
                                encode=encode, policy=policy)
        lookup = {f.id: f for f in inventory}
        recent = [lookup[i] for i in selected.recent_ids]
        middle = [lookup[i] for i in selected.middle_ids]
        # Shared renderer, but explicitly distinct live-note trust, not reviewed labels.
        native, _, candidate_counts = _messages(current, recent, middle,
                                                [SimpleNamespace(**n) for n in notes])
        candidate = _wire(native)
        tokens = count(candidate, candidate_counts)
        while tokens > max_input_tokens and middle:
            middle.pop(0)
            native, _, candidate_counts = _messages(current, recent, middle,
                                                    [SimpleNamespace(**n) for n in notes])
            candidate = _wire(native)
            tokens = count(candidate, candidate_counts)
        while tokens > max_input_tokens and len(recent) > 8:
            recent.pop(0)
            native, _, candidate_counts = _messages(current, recent, middle,
                                                    [SimpleNamespace(**n) for n in notes])
            candidate = _wire(native)
            tokens = count(candidate, candidate_counts)
    if tokens > max_input_tokens:
        raise ValueError("minimum live context exceeds token budget")
    records.append(vars(current).copy())
    # Older rows retain only a preselected sparse thumbnail; no old full images.
    recent_ids = {r["id"] for r in records[-policy.recent_frames:]}
    retained, last = [], None
    for row in records:
        if row["id"] in recent_ids:
            retained.append(row)
        elif ts - row["timestamp_ms"] <= policy.middle_horizon_ms and (
                last is None or row["timestamp_ms"] - last >= policy.middle_stride_ms):
            row["image"] = ""  # build_prompt chooses full images only from recent rows.
            retained.append(row)
            last = row["timestamp_ms"]
    new.update(records=retained, notes=notes, messages=candidate, counts=candidate_counts,
               sequence=int(current_id), last_ms=ts,
               generation=new.get("generation", 0) + int(compacted))
    return {"state": new, "messages": candidate, "input_tokens": tokens, "compacted": compacted,
            "pre_compaction_tokens": pre_compaction_tokens}
