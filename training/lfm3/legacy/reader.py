"""CPU-only immutable normalized registry index and lazy native-row adapter."""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path


from lumine_memory.render_policy import history_text

REVISION = "6f75ffb6c0ac670a51aa2b586315f0aa33156586"
COUNTS = {'train': 69480, 'validation': 7748}
POLICY = dict(recent_frames=8, middle_stride_ms=5000, middle_horizon_ms=60000,
              note_tokens=1024, input_tokens=7680, output_tokens=512,
              compact_every_ms=30000)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def records(path):
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                raise ValueError("blank registry line")
            yield json.loads(line)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def normalize_note(note):
    """Inline lineage becomes content-addressed child refs, not repeated blobs."""
    n = dict(note)
    if 'source_refs' in n:
        require(not n.get('source_notes'), 'mixed inline and normalized provenance')
        n.pop('source_notes',None)
        refs=n['source_refs']
        require(isinstance(refs,list) and all(set(r)=={'id','sha256'} for r in refs), 'normalized refs schema')
        require(len(refs)==len({r['id'] for r in refs}) and set(n.get('covers_note_ids',[]))=={r['id'] for r in refs},'normalized coverage')
        require(n.get('reviewed') is False and n.get('synthetic') is False and n.get('provenance')=='model-generated-annotation','unreviewed annotation trust')
        return n
    children = n.pop("source_notes", [])
    refs = []
    for child in children:
        if not isinstance(child, dict) or "text" not in child:
            raise ValueError("expected immutable inline source note")
        compact = normalize_note(child)
        refs.append({"id": child["id"], "sha256": digest(canonical(compact).encode())})
    n["source_refs"] = refs
    require(n.get("reviewed") is False and n.get("synthetic") is False and
            n.get("provenance") == "model-generated-annotation", "unreviewed annotation trust")
    require(set(n.get("covers_note_ids", [])) == {r["id"] for r in refs}, "lineage coverage")
    return n


def build_index(registry, output, actions=None, proof=None, expected=None):
    """Atomically publish index. No actions means metadata-only, unreadable for training."""
    registry, output = Path(registry), Path(output)
    expected = COUNTS if expected is None else expected
    require(not output.exists(), "index output already exists; use a fresh candidate path")
    sources = {n: filehash(registry / (n + ".jsonl"))
               for n in ("frames", "nodes", "frontier-events", "snapshots")}
    tmp = output.with_suffix(output.suffix + ".building")
    require(not tmp.exists(), "unfinished candidate exists")
    db = sqlite3.connect(tmp)
    try:
        db.executescript("""
        PRAGMA journal_mode=DELETE;
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE frames(id TEXT PRIMARY KEY,clip TEXT NOT NULL,split TEXT NOT NULL,
          idx INTEGER NOT NULL,ts INTEGER NOT NULL,payload TEXT NOT NULL,action TEXT,
          UNIQUE(clip,idx),UNIQUE(clip,split,ts));
        CREATE INDEX frames_time ON frames(clip,split,ts);
        CREATE TABLE nodes(id TEXT PRIMARY KEY,clip TEXT,split TEXT,ts INTEGER,
          hash TEXT NOT NULL,payload TEXT NOT NULL,provenance TEXT NOT NULL);
        CREATE TABLE events(id TEXT PRIMARY KEY,clip TEXT,split TEXT,ts INTEGER,payload TEXT);
        CREATE INDEX events_time ON events(clip,split,ts);
        CREATE TABLE targets(ordinal INTEGER PRIMARY KEY,id TEXT UNIQUE,split TEXT,payload TEXT);
        CREATE INDEX targets_split ON targets(split,ordinal);
        """)
        counts = Counter()
        for f in records(registry / "frames.jsonl"):
            require(f["split"] in expected, "gap/unknown split")
            require(f["id"] == f'{f["clip"]}/frame_{f["frame_index"]:08d}.png', "original frame ID")
            require(f["timestamp_ms"] == f["frame_index"] * 200 and
                    round(f["timestamp_s"] * 1000) == f["timestamp_ms"], "source time/index")
            db.execute("INSERT INTO frames VALUES(?,?,?,?,?,?,NULL)",
                       (f["id"], f["clip"], f["split"], f["frame_index"], f["timestamp_ms"], canonical(f)))
            counts[f["split"]] += 1
        require(dict(counts) == expected, "full source split coverage")
        for obj in records(registry / "nodes.jsonl"):
            require(obj["source_revision"] == REVISION, "node source revision")
            n = normalize_note(obj["note"])
            db.execute("INSERT INTO nodes VALUES(?,?,?,?,?,?,?)",
                       (n["id"],n["clip"],n["split"],n["timestamp_ms"],
                        digest(canonical(n).encode()),canonical(n),
                        canonical({k:v for k,v in obj.items() if k != "note"})))
        # Validate immutable child refs against standalone normalized children.
        for payload, in db.execute("SELECT payload FROM nodes"):
            n = json.loads(payload)
            for ref in n["source_refs"]:
                child = db.execute("SELECT hash,clip,split,ts FROM nodes WHERE id=?", (ref["id"],)).fetchone()
                require(child is not None and child[0] == ref["sha256"], "missing/stale child lineage")
                require(tuple(child[1:3]) == (n["clip"], n["split"]) and child[3] < n["timestamp_ms"],
                        "child causal split/time")
            evidence = n["evidence_frame_ids"]
            require(evidence and len(evidence) == len(set(evidence)), "note evidence")
            for fid in evidence:
                f = db.execute("SELECT clip,split,ts FROM frames WHERE id=?", (fid,)).fetchone()
                require(f is not None and tuple(f[:2]) == (n["clip"], n["split"])
                        and f[2] < n["timestamp_ms"], "future/cross-split/missing note evidence")
        for e in records(registry / "frontier-events.jsonl"):
            require(len(e["active_ids"]) == len(set(e["active_ids"])), "duplicate active note")
            for nid in e["active_ids"]:
                n = db.execute("SELECT clip,split,ts FROM nodes WHERE id=?", (nid,)).fetchone()
                require(n is not None and tuple(n[:2]) == (e["clip"],e["split"])
                        and n[2] <= e["timestamp_ms"], "invalid event note")
            db.execute("INSERT INTO events VALUES(?,?,?,?,?)",
                       (e["applied_id"],e["clip"],e["split"],e["timestamp_ms"],canonical(e)))
        cold = 0
        for ordinal, s in enumerate(records(registry / "snapshots.jsonl")):
            f = db.execute("SELECT clip,split,ts FROM frames WHERE id=?", (s["target_id"],)).fetchone()
            require(f is not None and (s["split"],s["timestamp_ms"]) == tuple(f[1:]), "target identity")
            require(s.get("note_trust") == "annotation", "snapshot annotation trust")
            event = db.execute("SELECT id,payload FROM events WHERE clip=? AND split=? AND ts<? AND ts>=? ORDER BY ts DESC,rowid DESC LIMIT 1",
                               (f[0],f[1],f[2],30200 if f[0]=="2026-10-30 11-42-21" and f[2]>=30200 else 0)).fetchone()
            require(s["event_id"] == (event[0] if event else None), "stale/noncausal snapshot")
            require(s["requires_notes"] is bool(event), "cold-start policy")
            require(s["cache_generation"] == (json.loads(event[1])["cache_generation"] if event else 0),
                    "snapshot generation")
            cold += not bool(event)
            db.execute("INSERT INTO targets VALUES(?,?,?,?)", (ordinal,s["target_id"],s["split"],canonical(s)))
        require(db.execute("SELECT COUNT(*) FROM targets").fetchone()[0] == sum(expected.values()), "no-drop targets")
        require(dict(db.execute("SELECT split,COUNT(*) FROM targets GROUP BY split")) == expected, "target splits")
        action_hash = None
        if actions is not None:
            require(proof is not None and Path(proof).is_file(), "full action proof required")
            p = json.loads(Path(proof).read_bytes())
            action_hash = filehash(actions)
            require(p.get("status") == "PASS" and p.get("revision") == REVISION and p.get("sha256") == action_hash and
                    p.get("counts") == expected and p.get("rows") == sum(expected.values()) and
                    p.get("all_action_hashes_verified") is True, "invalid full action proof")
            seen = set()
            for a in records(actions):
                key = (a["session"], a["frame_index"])
                require(key not in seen, "duplicate session/index action")
                seen.add(key)
                entry = db.execute("SELECT id,payload FROM frames WHERE clip=? AND idx=?",key).fetchone()
                require(entry is not None, "action identity absent")
                f = json.loads(entry[1])
                for k,v in f.items():
                    require(a.get(k) == v, "source action metadata mismatch " + k)
                require(a["source_revision"] == REVISION and
                        digest(a["action"].encode()) == f["action_sha256"], "action hash/revision")
                db.execute("UPDATE frames SET action=? WHERE id=?", (a["action"], entry[0]))
            require(len(seen) == sum(expected.values()), "incomplete action export")
            require(db.execute("SELECT COUNT(*) FROM frames WHERE action IS NULL").fetchone()[0] == 0,
                    "missing action join")
        require(all(filehash(registry/(n+".jsonl")) == h for n,h in sources.items()), "registry changed while indexing")
        report = dict(schema=1, source_revision=REVISION, counts=dict(counts), targets=sum(counts.values()),
                      cold_starts=cold, sources=sources, action_sha256=action_hash,
                      actions_verified=actions is not None, training_ready=False,
                      full_note_budget_pass=False, full_native_pixel_gate=False)
        db.execute("INSERT INTO meta VALUES('manifest',?)", (canonical(report),))
        db.commit()
        db.close()
        os.replace(tmp, output)
        return report
    except BaseException:
        db.close()
        if tmp.exists():
            tmp.unlink()
        raise


class NormalizedReader:
    """Connections reopen per PID and after pickle, safe for dataloader workers."""
    def __init__(self, index, policy=None):
        self.index = str(Path(index).resolve())
        self.policy = dict(POLICY if policy is None else policy)
        require(self.policy["note_tokens"] == 1024 and self.policy["input_tokens"] == 7680
                and self.policy["output_tokens"] == 512, "approved budgets immutable")
        self._db = None
        self._pid = None

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        return dict(self.__dict__, _db=None, _pid=None)

    @property
    def db(self):
        if self._pid != os.getpid() or self._db is None:
            if self._db is not None:
                self._db.close()
            self._db = sqlite3.connect(Path(self.index).as_uri() + "?mode=ro", uri=True)
            self._db.execute("PRAGMA query_only=ON")
            self._pid = os.getpid()
        return self._db

    @property
    def manifest(self):
        return json.loads(self.db.execute("SELECT value FROM meta WHERE key='manifest'").fetchone()[0])

    def ids(self, split):
        require(split in ("train","validation"), "unknown split")
        return [r[0] for r in self.db.execute("SELECT id FROM targets WHERE split=? ORDER BY ordinal",(split,))]

    def row(self, target_id):
        require(self.manifest["actions_verified"], "full source action export/proof not verified")
        s = self.db.execute("SELECT payload FROM targets WHERE id=?", (target_id,)).fetchone()
        if s is None:
            raise KeyError(target_id)
        snap = json.loads(s[0])
        raw = json.loads(self.db.execute("SELECT payload FROM frames WHERE id=?", (target_id,)).fetchone()[0])
        clip, split, ts = raw["clip"], raw["split"], raw["timestamp_ms"]
        event = None
        if snap["event_id"] is not None:
            event = json.loads(self.db.execute("SELECT payload FROM events WHERE id=?",(snap["event_id"],)).fetchone()[0])
            require((event["clip"],event["split"]) == (clip,split) and event["timestamp_ms"] < ts,
                    "noncausal event")
        needed = {target_id}
        provenance = {}
        def resolve(nid, stack=()):
            require(nid not in stack, "lineage cycle")
            row = self.db.execute("SELECT payload,provenance FROM nodes WHERE id=?", (nid,)).fetchone()
            require(row is not None, "missing node")
            n = json.loads(row[0])
            require((n["clip"],n["split"]) == (clip,split) and n["timestamp_ms"] < ts, "noncausal node")
            provenance[nid] = json.loads(row[1])
            needed.update(n["evidence_frame_ids"])
            children = []
            for ref in n.pop("source_refs"):
                child = self.db.execute("SELECT hash FROM nodes WHERE id=?", (ref["id"],)).fetchone()
                require(child is not None and child[0] == ref["sha256"], "stale child ref")
                children.append(resolve(ref["id"], (*stack,nid)))
            n["source_notes"] = children
            # Render-only projection. Exact raw text remains in immutable provenance.
            provenance[nid]["original_text"] = n["text"]
            n["text"] = history_text(n["text"])
            return n
        notes = [resolve(nid) for nid in event["active_ids"]] if event else []
        # Include exact horizon candidate inventory plus recent tail and all recursive
        # evidence. Extra evidence cannot alter the selector anchor outside horizon;
        # the last-N query preserves recent context even across sparse timestamps.
        needed.update(r[0] for r in self.db.execute(
            "SELECT id FROM frames WHERE clip=? AND split=? AND ts>=? AND ts<=?",
            (clip,split,ts-self.policy["middle_horizon_ms"],ts)))
        needed.update(r[0] for r in self.db.execute(
            "SELECT id FROM frames WHERE clip=? AND split=? AND ts<? ORDER BY ts DESC LIMIT ?",
            (clip,split,ts,self.policy["recent_frames"])))
        frames = []
        source_frames = {}
        for fid in sorted(needed):
            result = self.db.execute("SELECT payload,action FROM frames WHERE id=?", (fid,)).fetchone()
            require(result is not None, "missing evidence frame")
            f, action = json.loads(result[0]), result[1]
            require((f["clip"],f["split"]) == (clip,split) and f["timestamp_ms"] <= ts, "future/cross-split frame")
            require(fid == target_id or f["timestamp_ms"] < ts, "history target action leak")
            require(digest(action.encode()) == f["action_sha256"], "action index corruption")
            image = f'{clip}/frames/frame_{f["frame_index"]:08d}.png'
            frames.append(dict(id=fid,clip=clip,split=split,timestamp_ms=f["timestamp_ms"],
                               image=image,thumbnail="tiered/thumbnails/"+image,action=action))
            source_frames[fid] = f
        frames.sort(key=lambda f:(f["timestamp_ms"],f["id"]))
        return dict(source="lumine_tiered",target_id=target_id,frames=frames,notes=notes,
                    policy=self.policy,requires_notes=snap["requires_notes"],note_trust="annotation",
                    provisional=True,source_revision=REVISION,source_frames=source_frames,
                    node_provenance=provenance,normalized_snapshot=snap)


class RowTransform:
    """Pass as transform to VeOmni's registered mapping dataset over row-ID JSONL."""
    def __init__(self, index, processor, position_id_func, policy=None, image_root="."):
        self.reader = NormalizedReader(index,policy)
        self.processor, self.position_id_func = processor,position_id_func
        self.image_root = image_root

    def __call__(self, sample, **kwargs):
        from lumine_memory.transform import process_tiered_sample
        require(sample.get("source_revision") == REVISION, "row-ID source revision")
        row = self.reader.row(sample["target_id"])
        if "split" in sample:
            require(sample["split"] == row["normalized_snapshot"]["split"], "row-ID split")
        return process_tiered_sample(row,self.processor,self.position_id_func,
                                     max_seq_len=8192,image_root=self.image_root)


def write_row_ids(index, output):
    reader = NormalizedReader(index)
    require(reader.manifest["actions_verified"], "cannot publish training handles without actions")
    output = Path(output)
    output.mkdir(exist_ok=True)
    for split in ("train","validation"):
        with (output/(split+".jsonl")).open("x") as f:
            for tid in reader.ids(split):
                f.write(canonical(dict(target_id=tid,split=split,source_revision=REVISION))+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--registry",required=True)
    p.add_argument("--index",required=True)
    p.add_argument("--actions")
    p.add_argument("--proof")
    p.add_argument("--row-ids")
    a=p.parse_args()
    result=build_index(a.registry,a.index,a.actions,a.proof)
    if a.row_ids:
        write_row_ids(a.index,a.row_ids)
    print(json.dumps(result,indent=2))
