import json
import sqlite3
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'legacy'))
from reader import NormalizedReader, digest


def test_note_future_evidence_rejected(tmp_path):
    p = tmp_path / 'index.sqlite'
    db = sqlite3.connect(p)
    db.executescript('CREATE TABLE meta(key,value); CREATE TABLE targets(id,payload,split,ordinal); CREATE TABLE frames(id,payload,action,clip,split,ts); CREATE TABLE events(id,payload); CREATE TABLE nodes(id,payload,provenance,hash);')
    db.execute('INSERT INTO meta VALUES (?,?)', ('manifest', json.dumps(dict(actions_verified=True))))
    db.execute('INSERT INTO targets VALUES (?,?,?,?)', ('target',json.dumps(dict(event_id='event',requires_notes=True)), 'train',0))
    f = dict(clip='clip',split='train',timestamp_ms=100,frame_index=1, action_sha256=digest(b'w'))
    db.execute('INSERT INTO frames VALUES (?,?,?,?,?,?)', ('target',json.dumps(f),'w','clip','train',100))
    db.execute('INSERT INTO events VALUES (?,?)', ('event',json.dumps(dict(clip='clip',split='train',timestamp_ms=50,active_ids=['note']))))
    db.execute('INSERT INTO nodes VALUES (?,?,?,?)', ('note',json.dumps(dict(clip='clip',split='train',timestamp_ms=200,evidence_frame_ids=[],source_refs=[],text='future')), '{}','hash'))
    db.commit(); db.close()
    with pytest.raises(ValueError, match='noncausal node'):
        NormalizedReader(p).row('target')
