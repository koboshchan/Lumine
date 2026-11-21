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
