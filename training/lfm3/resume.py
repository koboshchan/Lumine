"""Fail-closed full-state resume checks, before loading any model weights."""
import json
import math
import os
from pathlib import Path

from action_tokens import ACTION_TOKENS


def validate_action_tokenizer(checkpoint, tokenizer):
    """Never register, resize or initialize anything on a resume path."""
    checkpoint = Path(checkpoint)
    expected = json.loads(Path(__file__).with_name('action-token-audit.json').read_text())
    audit_path = checkpoint / 'action-token-audit.json'
    if not audit_path.is_file():
        raise ValueError('Action-token resume requires checkpoint action-token-audit.json')
    audit = json.loads(audit_path.read_text())
    # Checkpoint audits historically contain the token mapping without a wrapper.
    audit = audit.get('tokens', audit)
    if set(audit) != set(ACTION_TOKENS) or len(tokenizer) != expected['vocab_after']:
        raise ValueError('Action-token resume mismatch: expected exactly 15 actions and vocab size 125046')
    for token in ACTION_TOKENS:
        token_id = expected['tokens'][token]['id']
        entry = tokenizer.added_tokens_decoder.get(token_id)
        if (audit[token].get('id') != token_id
                or tokenizer.convert_tokens_to_ids(token) != token_id
                or tokenizer.encode(token, add_special_tokens=False) != [token_id]
                or entry is None or str(entry) != token
                or not entry.special or entry.single_word != token.isalnum()
                or entry.lstrip or entry.rstrip or entry.normalized):
            raise ValueError(f'Action-token resume mismatch for {token!r}: expected token ID {token_id}')
    return audit


def validate_resume(checkpoint, config):
    checkpoint = Path(checkpoint)
    for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt',
                 'training-policy.json', 'config.json'):
        if not (checkpoint / name).is_file():
            raise ValueError('Incomplete resume checkpoint: missing ' + name)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rng_names = (['rng_state.pth'] if world_size == 1 else
                 [f'rng_state_{rank}.pth' for rank in range(world_size)])
    if any(not (checkpoint / name).is_file() for name in rng_names):
        raise ValueError('Incomplete resume checkpoint: missing RNG state or different world size')
    if json.loads((checkpoint / 'training-policy.json').read_text()) != config:
        raise ValueError('Checkpoint provenance does not match training config (subset/seed/data policy included)')
    return json.loads((checkpoint / 'trainer_state.json').read_text())


def resume_settings(config, targets, max_steps, world_size):
    return dict(targets=targets, world_size=world_size, seed=42, data_seed=42,
                microbatch=1, gradient_accumulation_steps=config['gradient_accumulation_steps'],
                max_steps=max_steps, epochs=config['epochs'])


def validate_data_order(checkpoint, settings):
    """Trainer skips consumed batches; changing epoch length would skip wrong data."""
    checkpoint = Path(checkpoint)
    saved = checkpoint / 'resume-settings.json'
    if saved.exists():
        if json.loads(saved.read_text()) != settings:
            raise ValueError('Resume data order/schedule differs (targets, world size, seeds or max steps)')
    elif settings['world_size'] != 1:
        raise ValueError('Legacy checkpoint has no world-size provenance; only single-GPU resume is supported')
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    steps_per_epoch = math.ceil(math.ceil(settings['targets'] / settings['world_size']) /
                                settings['gradient_accumulation_steps'])
    planned = settings['max_steps'] if settings['max_steps'] > 0 else math.ceil(
        steps_per_epoch * settings['epochs'])
    if state['max_steps'] != planned:
        raise ValueError('Resume max steps differs from saved scheduler; keep original --max-steps/epochs')
    # Earlier checkpoints have no settings file. Policy still pins dataset, target
    # subset/seed and bootstrap. Trainer's original seeds are fixed at 42.
    if state.get('train_batch_size', 1) != settings['microbatch']:
        raise ValueError('Resume microbatch differs from checkpoint')
