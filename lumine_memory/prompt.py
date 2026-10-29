"""Deterministic causal tier selection. No ML, GPU, network or server imports."""
from dataclasses import dataclass, replace
from typing import Callable, Sequence

SCHEMA_VERSION = "lumine-tiered-v1"


@dataclass(frozen=True)
class Frame:
    id: str
    clip: str
    split: str
    timestamp_ms: int
    image: str
    thumbnail: str
    action: str


@dataclass(frozen=True)
class MemoryNote:
    id: str
    clip: str
    split: str
    timestamp_ms: int
    text: str
    evidence_frame_ids: tuple[str, ...]
    reviewed: bool = False
    synthetic: bool = False


@dataclass(frozen=True)
class Policy:
    recent_frames: int = 16
    middle_stride_ms: int = 2000
    middle_horizon_ms: int = 300000
    note_tokens: int = 2048
    input_tokens: int = 30000
    output_tokens: int = 512
    compact_every_ms: int = 30000

    def __post_init__(self):
        if self.recent_frames not in (8, 16):
            raise ValueError("recent_frames must be 8 or 16")
        if min(self.middle_stride_ms, self.middle_horizon_ms, self.note_tokens,
               self.input_tokens, self.output_tokens, self.compact_every_ms) <= 0:
            raise ValueError("budgets and periods must be positive")
        if self.input_tokens + self.output_tokens >= 32000:
            raise ValueError("combined context must be below 32000")


@dataclass(frozen=True)
class Prompt:
    messages: tuple[dict, ...]
    images: tuple[str, ...]
    image_tokens: tuple[int, ...]
    recent_ids: tuple[str, ...]
    middle_ids: tuple[str, ...]
    note_ids: tuple[str, ...]
    token_count: int
    discarded_middle: int
    discarded_recent: int
    cache_segment: int
    schema_version: str = SCHEMA_VERSION


def render_text(messages: Sequence[dict], apply_template: Callable, image_tokens: Sequence[int],
                *, generation_prompt: bool = True) -> str:
    """Render native HF template and expand image markers for exact tokenizer counts."""
    text = apply_template(list(messages), tokenize=False, add_generation_prompt=generation_prompt)
    parts = text.split("<|image_pad|>")
    if len(parts) != len(image_tokens) + 1:
        raise ValueError("native template image marker count mismatch")
    return parts[0] + "".join("<|image_pad|>" * n + tail
                             for n, tail in zip(image_tokens, parts[1:], strict=True))


def compact_notes(notes: Sequence[MemoryNote], summary: MemoryNote) -> tuple[MemoryNote, ...]:
    """Replace covered segments with a reviewed VLM summary; caller resets KV cache.

    This is a state operation, not an automatic summarizer. Uncovered newer notes
    remain append-only. No silent dropping of facts when a note budget is exceeded.
    """
    covered = [n for n in notes if n.timestamp_ms <= summary.timestamp_ms]
    if not summary.reviewed or not summary.text.strip() or not covered:
        raise ValueError("compaction requires a reviewed nonempty summary and covered notes")
    if any((n.clip, n.split) != (summary.clip, summary.split) for n in notes):
        raise ValueError("compaction cannot cross clip/split")
    evidence = {f for n in covered for f in n.evidence_frame_ids}
    if not evidence.issubset(summary.evidence_frame_ids):
        raise ValueError("summary must preserve evidence provenance")
    return (summary, *(n for n in notes if n.timestamp_ms > summary.timestamp_ms))


def _messages(current, recent, middle, notes):
    messages = []
    if notes:
        messages.append({"role": "user", "content": [{"type": "text", "text":
            "<|memory|>\n" + "\n".join(f"[{n.timestamp_ms}ms] {n.text}" for n in notes) + "\n</|memory|>"}]})
    images, counts = [], []
    for tier, rows, count in (("middle", middle, 60), ("recent", recent, 920),
                              ("current", [current], 920)):
        for row in rows:
            path = row.thumbnail if tier == "middle" else row.image
            messages.append({"role": "user", "content": [
                {"type": "text", "text": f"[{tier} {row.timestamp_ms}ms]"},
                {"type": "image", "image": path}]})
            images.append(path)
            counts.append(count)
            if tier != "current":
                messages.append({"role": "assistant", "content": row.action})
    return messages, images, counts


def build_prompt(frames: Sequence[Frame], current: Frame, notes: Sequence[MemoryNote], *,
                 apply_template: Callable, encode: Callable, policy: Policy = Policy(),
                 allow_synthetic: bool = False) -> Prompt:
    """Build shared prompt. encode(text) returns raw IDs without added special tokens.

    frames is the full authoritative same-clip/split observation inventory, excluding
    gap frames. current.action is never rendered. URLs, paths and data URIs are
    opaque references; consumers load the selected image representations.
    """
    if current.split not in ("train", "validation"):
        raise ValueError("current must be train or validation, never gap")
    ids = [f.id for f in frames]
    if len(set(ids)) != len(ids) or current.id not in ids:
        raise ValueError("inventory must have unique IDs and contain current")
    lookup = {f.id: f for f in frames}
    if lookup[current.id] != current:
        raise ValueError("current differs from authoritative inventory")
    past = sorted((f for f in frames if (f.clip, f.split) == (current.clip, current.split)
                   and f.timestamp_ms < current.timestamp_ms), key=lambda f: (f.timestamp_ms, f.id))
    if len({f.timestamp_ms for f in past}) != len(past):
        raise ValueError("duplicate timestamps within clip/split")
    recent = past[-policy.recent_frames:]
    cutoff = recent[0].timestamp_ms if recent else current.timestamp_ms
    middle, last = [], None
    for f in past:
        if f.timestamp_ms >= cutoff or f.timestamp_ms < current.timestamp_ms - policy.middle_horizon_ms:
            continue
        if last is None or f.timestamp_ms - last >= policy.middle_stride_ms:
            middle.append(f)
            last = f.timestamp_ms
    active = sorted((n for n in notes if (n.clip, n.split) == (current.clip, current.split)
                     and n.timestamp_ms < current.timestamp_ms), key=lambda n: (n.timestamp_ms, n.id))
    for note in active:
        if not note.reviewed or not note.text.strip() or not note.evidence_frame_ids:
            raise ValueError("notes require reviewed nonempty evidence-linked annotations")
        if note.synthetic and not allow_synthetic:
            raise ValueError("synthetic notes forbidden for real data")
        for fid in note.evidence_frame_ids:
            f = lookup.get(fid)
            if f is None or (f.clip, f.split) != (current.clip, current.split) or f.timestamp_ms > note.timestamp_ms:
                raise ValueError("note evidence is missing, future-looking, gap or cross-split")
    note_text = "\n".join(f"[{n.timestamp_ms}ms] {n.text}" for n in active)
    if len(encode(note_text)) > policy.note_tokens:
        raise ValueError("notes exceed budget; reviewed compaction required")
    dm = dr = 0
    while True:
        messages, images, counts = _messages(current, recent, middle, active)
        tokens = len(encode(render_text(messages, apply_template, counts)))
        if tokens <= policy.input_tokens:
            break
        if middle:
            middle.pop(0)
            dm += 1
        elif len(recent) > 8:
            recent.pop(0)
            dr += 1
        else:
            raise ValueError("minimum recent context exceeds token budget")
    return Prompt(tuple(messages), tuple(images), tuple(counts), tuple(f.id for f in recent),
                  tuple(f.id for f in middle), tuple(n.id for n in active), tokens, dm, dr,
                  current.timestamp_ms // policy.compact_every_ms)
