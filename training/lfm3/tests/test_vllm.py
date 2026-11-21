"""Optional backend tests require no vLLM installation or accelerator."""
import copy
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import encode_prompt
from serve_vllm import FrameHashes, PromptBuilder
from train import ActionCollator


def test_content_hashes_bounded_and_resolution_specific():
    cache = FrameHashes(2)
    a = Image.new('RGB', (640, 360), 'green')
    assert cache.get(a) == cache.get(a.copy())
    assert cache.get(a) != cache.get(a.resize((320, 180)))
    assert len(cache.items) == 2
    assert cache.get(a) != cache.get(Image.new('RGB', a.size, 'blue'))
    assert len(cache.items) == 2


def test_native_prompt_budget_and_pixels():
    import torch
    from transformers import AutoProcessor
    config = json.loads((ROOT / 'config.json').read_text())
    processor = AutoProcessor.from_pretrained(config['model'], revision=config['model_revision'],
                                            trust_remote_code=True)
    processor.chat_template = (ROOT / 'chat_template.jinja').read_text()
    row = lambda ts: dict(image=Image.new('RGB', (640, 360), (10, 110, int(ts))),
                          action='<|action_start|>0 0 0 ;  ;  ;  ;  ;  ; <|action_end|>', timestamp_s=ts)
    sample = dict(current=row(10), recent=[row(i) for i in range(8)], middle=[], notes=[])
    config['history_tokens'] = 600  # Force trimming and test the unchanged rule.
    source = copy.deepcopy(sample)
    def render(raw):
        batch = processor(text=raw['prompt'], images=raw['multi_modal_data']['image'],
                          add_special_tokens=False, return_tensors='pt')
        return {'prompt_token_ids': batch['input_ids'][0].tolist(), 'batch': batch}
    builder = PromptBuilder(processor, config, render)
    raw, rendered, effective = builder.prepare(sample)
    expected = encode_prompt(processor, config, sample)
    for name in ('input_ids', 'pixel_values', 'image_grid_thw', 'attention_mask'):
        assert torch.equal(rendered['batch'][name], expected[name]), name
    assert len(effective['recent']) < len(source['recent'])
    assert len(sample['recent']) == 8
    assert raw['prompt'].endswith('<|im_start|>assistant\n')
    assert raw['prompt'].count('[recent ') == len(effective['recent'])
    assert len(raw['multi_modal_uuids']['image']) == len(effective['recent']) + 1
    assert raw == builder.raw(effective)


@pytest.mark.parametrize('extra', [
    {'bootstrap_prob': float('nan')}, {'bootstrap_prob': -0.1},
    {'bootstrap_prob': True}, {'bootstrap_seed': True},
    {'bootstrap_seed': '42'},
])
def test_checkpoint_rejects_invalid_training_provenance(tmp_path, extra):
    from serve import load_checkpoint_processor
    config = json.loads((ROOT / 'config.json').read_text())
    config.update(extra)
    (tmp_path / 'training-policy.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='bootstrap'):
        load_checkpoint_processor(tmp_path)


def test_checkpoint_accepts_valid_provenance_before_processor(tmp_path, monkeypatch):
    from serve import load_checkpoint_processor
    import transformers
    config = json.loads((ROOT / 'config.json').read_text())
    config.update(bootstrap_prob=0.1, bootstrap_seed=42)
    (tmp_path / 'training-policy.json').write_text(json.dumps(config))
    class ProcessorReached(Exception):
        pass
    def reached(*args, **kwargs):
        raise ProcessorReached()
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', reached)
    with pytest.raises(ProcessorReached):
        load_checkpoint_processor(tmp_path)


def test_fast_budget_matches_stock_and_decode_reused():
    import io
    from serve import load_checkpoint_processor
    from transformers import AutoProcessor
    config = json.loads((ROOT / 'config.json').read_text())
    config['history_tokens'] = 600
    p = AutoProcessor.from_pretrained(config['model'], revision=config['model_revision'], trust_remote_code=True)
    p.chat_template = (ROOT / 'chat_template.jinja').read_text()
    def row(ts):
        out = io.BytesIO()
        Image.new('RGB', (640, 360), (5, 20, ts)).save(out, 'PNG')
        return dict(image={'bytes': out.getvalue()}, action='0', timestamp_s=ts)
    sample = dict(current=row(10), recent=[row(i) for i in range(8)], middle=[], notes=[])
    fast = PromptBuilder(p, config, lambda raw: None, fast=True)
    raw, rendered, effective = fast.prepare(sample)
    stock = encode_prompt(p, config, sample)
    assert rendered['prompt_token_ids'] == stock['input_ids'][0].tolist()
    misses = fast.decoded.misses
    assert fast.prepare(sample)[1] == rendered
    assert fast.decoded.misses == misses == 9
    assert len(sample['recent']) == 8


def test_exact_pixel_cache_mixed_sizes_kwargs_and_bound():
    import torch
    from transformers.models.qwen2_vl import Qwen2VLImageProcessor
    from frame_cache import PixelFrames
    p = Qwen2VLImageProcessor(size={'shortest_edge': 65536, 'longest_edge': 230400}, patch_size=16)
    images = [Image.new('RGB', size, color) for size, color in
              [((640, 360), 'red'), ((320, 180), 'green'), ((640, 360), 'blue')]]
    expected = p(images, return_tensors='pt')
    cache = PixelFrames()
    cache.install(p)
    actual = p(images, return_tensors='pt')
    for name in expected:
        assert torch.equal(expected[name], actual[name])
    assert cache.misses == 3
    p([im.copy() for im in images], return_tensors='pt')
    assert cache.hits == 3
    actual['pixel_values'].zero_()
    assert torch.equal(p(images, return_tensors='pt')['pixel_values'], expected['pixel_values'])
    p(images, return_tensors='pt', do_normalize=False)
    assert cache.misses == 6
    assert cache.bytes <= cache.max_bytes
    small = PixelFrames(max_bytes=1)
    other = Qwen2VLImageProcessor()
    small.install(other)
    other(images[:1], return_tensors='pt')
    assert small.bytes == 0 and not small.items
