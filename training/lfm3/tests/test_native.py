"""CPU acceptance against the exact upstream processor and a tiny native LM."""
import importlib.util
import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import ActionCollator


@pytest.fixture(scope='module')
def setup():
    from transformers import AutoProcessor
    c = json.loads((ROOT / 'config.json').read_text())
    p = AutoProcessor.from_pretrained(c['model'], revision=c['model_revision'], trust_remote_code=True)
    p.chat_template = (ROOT / 'chat_template.jinja').read_text()
    return p, c


def sample():
    from PIL import Image
    row = lambda action, ts: dict(image=Image.new('RGB', (640, 360), 'green'), action=action, timestamp_s=ts)
    return dict(current=row('w <mouse_left>', 1.0), recent=[row('s <mouse_right>', .5)], middle=[],
                notes=[dict(timestamp_ms=100, text='A hill is ahead.')])


def test_mask_and_images(setup):
    p, c = setup
    collator = ActionCollator(p, c)
    s = sample()
    b = collator([s])
    supervised = p.tokenizer.decode(b['labels'][b['labels'] != -100].tolist())
    assert supervised.startswith(s['current']['action'])
    assert 'Previous action' not in supervised and 'hill' not in supervised
    assert 'mouse_right' not in supervised and '<think>' not in supervised
    rendered = p.apply_chat_template(collator.messages(s), tokenize=False, add_generation_prompt=False)
    assert '<think>' not in rendered and '</think>' not in rendered
    generated = p.apply_chat_template(collator.messages(s)[:-1], tokenize=False, add_generation_prompt=True)
    assert '<think>' not in generated
    assert int((b['labels'] != -100).sum()) < b['input_ids'].numel() // 2
    image_id = p.tokenizer.convert_tokens_to_ids('<|image|>')
    actual = int((b['input_ids'] == image_id).sum())
    expected = int((b['image_grid_thw'].prod(-1) // (p.image_processor.merge_size ** 2)).sum())
    assert actual == expected and actual > 0
    assert b['pixel_values'].numel() > 0
    assert (b['labels'][b['input_ids'] == image_id] == -100).all()


def test_tiny_native_backward(setup):
    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    p, c = setup
    cfg = AutoConfig.from_pretrained(c['model'], revision=c['model_revision'], trust_remote_code=True)
    # Use the actual remote LFM3 text decoder, not an unrelated architecture.
    text = cfg.text_config
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_hidden_layers = 2
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    if hasattr(text, 'head_dim'): text.head_dim = 16
    text.layer_types = ['full_attention', 'full_attention']
    cls = get_class_from_dynamic_module('modeling_lfm3.Lfm3ForCausalLM', c['model'], revision=c['model_revision'])
    model = cls(text).float()
    ids = p.tokenizer('w <mouse_left>', return_tensors='pt')['input_ids']
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        opt.zero_grad()
        loss = model(input_ids=ids, labels=ids).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert any(x.grad is not None and x.grad.abs().sum() > 0 for x in model.parameters())
        opt.step()


def test_no_empty_thinking_branches():
    text = (ROOT / 'chat_template.jinja').read_text()
    assert '<think>\\n\\n</think>' not in text
    assert 'generation' in text
