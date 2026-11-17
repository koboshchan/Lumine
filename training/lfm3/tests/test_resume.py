"""Actual CPU Trainer interruption/resume with native action15 tokenizer."""
import copy
import json
from pathlib import Path
import random
import sys
import tarfile

import numpy as np
import pytest
import torch
from transformers import (AutoProcessor, GPT2Config, GPT2LMHeadModel,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from action_tokens import initialize_action_tokens, save_action_tokens
from checkpoint_io import restore
from resume import (validate_action_tokenizer, validate_resume, resume_settings,
                    validate_data_order)
from train import load_training_components, select_targets
from test_checkpoint_io import API


class Rows:
    def __init__(self):
        self.ids = list(range(12))
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        # Distinct examples, so incorrectly skipping or reshuffling changes weights.
        ids = [125031, 10 + self.ids[i], 125040, 125032]
        return dict(input_ids=ids, labels=ids)


@pytest.fixture
def token_model():
    c = json.loads((ROOT / 'config.json').read_text())
    p = AutoProcessor.from_pretrained(c['model'], revision=c['model_revision'], trust_remote_code=True)
    set_seed(42)
    m = GPT2LMHeadModel(GPT2Config(vocab_size=len(p.tokenizer), n_embd=8, n_layer=1,
                                  n_head=1, n_positions=16))
    audit = initialize_action_tokens(p, m)
    return p, m, audit


def test_full_state_resume_tokens_and_epoch(tmp_path, token_model, monkeypatch):
    p, m, audit = token_model
    c = json.loads((ROOT / 'config.json').read_text())
    c.update(action_special_tokens=True, subset=8, subset_seed=37,
             bootstrap_prob=.1, bootstrap_seed=37, gradient_accumulation_steps=1)
    rows = select_targets(Rows(), 8, 37)
    settings = resume_settings(c, len(rows), 4, 1)
    initial = copy.deepcopy(m.state_dict())

    class Save(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            cp = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            p.save_pretrained(cp)
            save_action_tokens(cp, audit)
            (cp / 'training-policy.json').write_text(json.dumps(c))
            (cp / 'resume-settings.json').write_text(json.dumps(settings))
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_save = True
                if 'interrupted' in args.output_dir:
                    control.should_training_stop = True

    def trainer(model, out):
        return Trainer(model=model, train_dataset=rows, callbacks=[Save()],
            args=TrainingArguments(output_dir=str(out), use_cpu=True, max_steps=4,
                per_device_train_batch_size=1, gradient_accumulation_steps=1,
                save_steps=100, learning_rate=1e-3, seed=42, data_seed=42,
                ignore_data_skip=False, report_to='none', dataloader_pin_memory=False))

    uninterrupted = trainer(m, tmp_path / 'reference')
    uninterrupted.train()
    reference = copy.deepcopy(m.state_dict())
    # Start again from the exact same initialization, then interrupt mid-epoch.
    m.load_state_dict(initial)
    interrupted = trainer(m, tmp_path / 'interrupted')
    assert interrupted.train().global_step == 2
    cp = tmp_path / 'interrupted' / 'checkpoint-2'
    weights = GPT2LMHeadModel.from_pretrained(cp).get_input_embeddings().weight.detach().clone()
    assert not torch.equal(weights, initial['transformer.wte.weight'])
    assert validate_resume(cp, c)['global_step'] == 2
    validate_data_order(cp, settings)
    assert validate_action_tokenizer(cp, p.tokenizer) == audit

    # Any registration, resize or mean-init during resume is a bug.
    def forbidden(*args, **kwargs):
        raise AssertionError('resume must not reinitialize/resize/register action tokens')
    import action_tokens
    monkeypatch.setattr(action_tokens, 'initialize_action_tokens', forbidden)
    monkeypatch.setattr(action_tokens, 'register_action_tokens', forbidden)
    monkeypatch.setattr(GPT2LMHeadModel, 'resize_token_embeddings', forbidden)
    restored_p, restored_m, restored_audit = load_training_components(
        c, str(cp), True, AutoProcessor, GPT2LMHeadModel, torch.float32)
    assert restored_audit == audit
    assert len(restored_p.tokenizer) == 125046
    assert torch.equal(restored_m.get_input_embeddings().weight, weights)
    for token, entry in audit.items():
        assert restored_p.tokenizer.encode(token, add_special_tokens=False) == [entry['id']]
    # Perturb all RNG streams. Trainer must restore them along with optimizer and LR.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    resumed = trainer(restored_m, tmp_path / 'continued')
    assert resumed.train(resume_from_checkpoint=str(cp)).global_step == 4
    assert resumed.state.epoch == uninterrupted.state.epoch
    for name, value in reference.items():
        assert torch.equal(resumed.model.state_dict()[name], value), name
    assert resumed.lr_scheduler.state_dict() == uninterrupted.lr_scheduler.state_dict()
    for a, b in zip(resumed.optimizer.state.values(), uninterrupted.optimizer.state.values()):
        for key in a:
            if torch.is_tensor(a[key]):
                assert torch.equal(a[key], b[key])
            else:
                assert a[key] == b[key]

    # All three archive/latest transports feed the same validated tokenized checkpoint.
    assert restore('latest', cp.parent, None) == str(cp)
    archive = tmp_path / 'checkpoint-2.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(cp, arcname=cp.name)
    local = restore(str(archive), tmp_path / 'local', None)
    api = API(tmp_path / 'remote')
    remote = api.root / 'user' / 'bucket' / 'run'
    remote.mkdir(parents=True)
    import shutil
    shutil.copy(archive, remote / archive.name)
    bucket = restore('hf://buckets/user/bucket/run/checkpoint-2.tar', tmp_path / 'bucket', api)
    for path in (local, bucket):
        validate_resume(path, c)
        rp = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        validate_action_tokenizer(path, rp.tokenizer)
        assert torch.equal(GPT2LMHeadModel.from_pretrained(path).get_input_embeddings().weight, weights)

    with pytest.raises(ValueError, match='provenance'):
        validate_resume(cp, dict(c, subset_seed=38))
    for key, value in [('targets', 7), ('world_size', 2), ('data_seed', 43), ('max_steps', 5)]:
        with pytest.raises(ValueError, match='data order/schedule'):
            validate_data_order(cp, dict(settings, **{key: value}))
    (cp / 'resume-settings.json').unlink()
    with pytest.raises(ValueError, match='max steps'):
        validate_data_order(cp, dict(settings, max_steps=5))
    (cp / 'rng_state.pth').unlink()
    with pytest.raises(ValueError, match='RNG'):
        validate_resume(cp, c)


@pytest.mark.parametrize('mismatch', ['id', 'missing', 'vocab', 'tokenizer_id', 'boundary'])
def test_resume_rejects_mismatch(tmp_path, token_model, mismatch):
    p, m, audit = token_model
    if mismatch == 'id':
        audit['LB']['id'] += 1
    elif mismatch == 'missing':
        del audit['Caps']
    elif mismatch == 'vocab':
        p.tokenizer.add_tokens(['extra-unrelated-resume-token'])
    p.save_pretrained(tmp_path)
    save_action_tokens(tmp_path, audit)
    if mismatch in ('tokenizer_id', 'boundary'):
        raw_path = tmp_path / 'tokenizer.json'
        raw = json.loads(raw_path.read_text())
        tokens = {t['content']: t for t in raw['added_tokens']}
        cfg_path = tmp_path / 'tokenizer_config.json'
        cfg = json.loads(cfg_path.read_text())
        decoder = cfg.get('added_tokens_decoder', {})
        if mismatch == 'tokenizer_id':
            a, b = tokens['Caps'], tokens['Nine']
            ia, ib = str(a['id']), str(b['id'])
            ai, bi = raw['added_tokens'].index(a), raw['added_tokens'].index(b)
            a['id'], b['id'] = b['id'], a['id']
            raw['added_tokens'][ai], raw['added_tokens'][bi] = b, a
            if decoder:
                decoder[ia], decoder[ib] = decoder[ib], decoder[ia]
        else:
            tokens['Caps']['single_word'] = False
            if decoder:
                decoder[str(tokens['Caps']['id'])]['single_word'] = False
        raw_path.write_text(json.dumps(raw))
        cfg_path.write_text(json.dumps(cfg))
    restored = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=True)
    with pytest.raises(ValueError, match='resume mismatch'):
        validate_action_tokenizer(tmp_path, restored.tokenizer)
