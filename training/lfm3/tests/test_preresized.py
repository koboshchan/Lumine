"""Small CPU checks for the embedded pre-resized Parquet representation."""
import io
import sys
from pathlib import Path
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import prepared_image, ActionCollator


def encoded(size):
    image = Image.new('RGB', size, 'green')
    data = io.BytesIO()
    image.save(data, format='PNG')
    return dict(bytes=data.getvalue(), path=None)


def test_decode_never_resizes(monkeypatch):
    row = dict(image=encoded((640, 360)), thumbnail=encoded((320, 180)))
    def forbidden(*args, **kwargs):
        raise AssertionError('Training must not resize prepared frames')
    monkeypatch.setattr(Image.Image, 'resize', forbidden)
    assert prepared_image(row, 640).size == (640, 360)
    assert prepared_image(row, 320).size == (320, 180)


def test_full_resolution_rejected():
    with pytest.raises(ValueError, match='pre-resized'):
        prepared_image(dict(image=encoded((1280, 720))), 640)


def test_collator_uses_independent_thumbnail(monkeypatch):
    row = dict(image=encoded((640, 360)), thumbnail=encoded((320, 180)),
               action='w', timestamp_s=1.0)
    def forbidden(*args, **kwargs):
        raise AssertionError('Unexpected resizing')
    monkeypatch.setattr(Image.Image, 'resize', forbidden)
    c = ActionCollator(None, dict(frame_width=640, thumbnail_width=320))
    messages = c.messages(dict(current=row, recent=[row], middle=[row], notes=[]))
    images = [part['image'] for m in messages if isinstance(m['content'], list)
              for part in m['content'] if part['type'] == 'image']
    assert [im.size for im in images] == [(320, 180), (640, 360), (640, 360)]


from prepare import verify_resized_shards
import hashlib


def resized_fixture(tmp_path):
    shard = tmp_path / 'data' / 'train.parquet'
    shard.parent.mkdir()
    shard.write_bytes(b'resized')
    old = dict(file='data/train.parquet', sha256='original', bytes=20)
    new = dict(file=old['file'], source_sha256='original', source_bytes=20,
               sha256=hashlib.sha256(shard.read_bytes()).hexdigest(), bytes=7)
    return shard, old, new


def test_resized_lineage_verified(tmp_path):
    shard, old, new = resized_fixture(tmp_path)
    verify_resized_shards(tmp_path, dict(shards=[new]), [old])


def test_resized_bytes_tamper_rejected(tmp_path):
    shard, old, new = resized_fixture(tmp_path)
    shard.write_bytes(b'changed')
    with pytest.raises(ValueError, match='shard hash'):
        verify_resized_shards(tmp_path, dict(shards=[new]), [old])


def test_wrong_source_lineage_rejected(tmp_path):
    shard, old, new = resized_fixture(tmp_path)
    new['source_sha256'] = 'different'
    with pytest.raises(ValueError, match='immutable source'):
        verify_resized_shards(tmp_path, dict(shards=[new]), [old])


def test_duplicate_shard_rejected(tmp_path):
    shard, old, new = resized_fixture(tmp_path)
    other = dict(old, file='data/other.parquet')
    with pytest.raises(ValueError, match='Duplicate'):
        verify_resized_shards(tmp_path, dict(shards=[new, new]), [old, other])
