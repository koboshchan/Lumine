"""CPU checks for target selection, idle bootstrap parity, and final optimizer saves."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import (select_targets, bootstrap_sample, final_checkpoint_control,
                   ActionCollator, BOOTSTRAP_ACTION)


class Targets:
    def __init__(self):
        self.ids = list(range(5000))
        self.rows = object()
        self.reader = object()
        self.lookup = object()
    def __len__(self):
        return len(self.ids)


def test_subset_deterministic_keeps_context():
    a, b, c = Targets(), Targets(), Targets()
    full = a.rows, a.reader, a.lookup
    select_targets(a, 1000, 42)
    select_targets(b, 1000, 42)
    select_targets(c, 1000, 43)
    assert a.ids == b.ids and a.ids != c.ids
    assert len(set(a.ids)) == 1000 and a.ids == sorted(a.ids)
    assert (a.rows, a.reader, a.lookup) == full
    full = Targets()
    assert select_targets(full, 0, 99).ids == list(range(5000))
    with pytest.raises(ValueError):
        select_targets(Targets(), 5001, 42)
    with pytest.raises(ValueError):
        select_targets(Targets(), -1, 42)


def test_bootstrap_stationary_proxy_matches_server_action():
    from PIL import Image
    row = dict(image=Image.new('RGB', (640, 360)), action='target', timestamp_s=93.4)
    sample = dict(current=row, recent=[dict(row, action='moving')], middle=[],
                  notes=[dict(timestamp_ms=1, text='old memory')])
    augmented = bootstrap_sample(sample, 8)
    assert len(augmented['recent']) == 8
    assert augmented['middle'] == augmented['notes'] == []
    assert augmented['current']['action'] == 'target'
    assert augmented['current']['timestamp_s'] == pytest.approx(.8)
    assert [r['timestamp_s'] for r in augmented['recent']] == pytest.approx([i*.1 for i in range(8)])
    assert all(r['action'] == BOOTSTRAP_ACTION and r['image'] is row['image']
               for r in augmented['recent'])
    assert sample['recent'][0]['action'] == 'moving' and row['timestamp_s'] == 93.4
    assert BOOTSTRAP_ACTION in (ROOT/'serve.py').read_text()


def test_final_checkpoint_on_non_cadence_step():
    from transformers import Trainer, TrainerCallback, TrainingArguments, GPT2Config, GPT2LMHeadModel
    import torch
    import tempfile
    class FinalSave(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            return final_checkpoint_control(state, control)
    class Rows:
        def __len__(self): return 8
        def __getitem__(self, i): return {'input_ids': [1, 2, 3], 'labels': [1, 2, 3]}
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(model=GPT2LMHeadModel(GPT2Config(vocab_size=8, n_embd=8, n_layer=1, n_head=1)),
            args=TrainingArguments(output_dir=tmp, use_cpu=True, max_steps=3,
                per_device_train_batch_size=1, gradient_accumulation_steps=2,
                save_steps=100, report_to='none', dataloader_pin_memory=False),
            train_dataset=Rows(), callbacks=[FinalSave()])
        result = trainer.train()
        assert result.global_step == 3
        cp = Path(tmp)/'checkpoint-3'
        assert all((cp/f).exists() for f in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt'))
        assert not (Path(tmp)/'checkpoint-2').exists()


def test_bootstrap_native_assistant_mask():
    from transformers import AutoProcessor
    from PIL import Image
    c = json.loads((ROOT/'config.json').read_text())
    p = AutoProcessor.from_pretrained(c['model'], revision=c['model_revision'], trust_remote_code=True)
    p.chat_template = (ROOT/'chat_template.jinja').read_text()
    row = dict(image=Image.new('RGB', (640, 360)), action='<|action_start|>0 0 0; W; W; W; W; W; W<|action_end|>', timestamp_s=5.)
    sample = bootstrap_sample(dict(current=row, recent=[], middle=[], notes=[]), 8)
    batch = ActionCollator(p, c)([sample])
    supervised = p.tokenizer.decode(batch['labels'][batch['labels'] != -100].tolist())
    assert supervised.startswith(row['action'])
    assert BOOTSTRAP_ACTION not in supervised
