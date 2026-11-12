"""Full-parameter LFM3 VLM SFT. No allocation, serving or background controllers."""
import argparse
import io
import json
import os
import random
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM = "Predict the next game input. Reply with only the action string."


# Exact server bootstrap spelling, including the double spaces between separators.
BOOTSTRAP_ACTION = '<|action_start|>0 0 0 ;  ;  ;  ;  ;  ; <|action_end|>'


def select_targets(dataset, count, seed):
    """Select targets only; leave the full source rows and causal reader untouched."""
    if count < 0:
        raise ValueError('subset must be nonnegative (0 means full training set)')
    if count > len(dataset):
        raise ValueError('subset exceeds the number of training targets')
    if count:
        dataset.ids = [dataset.ids[i] for i in sorted(
            random.Random(seed).sample(range(len(dataset)), count))]
    return dataset


def bootstrap_sample(sample, recent_frames):
    """Synthetic idle startup: repeat the current image, never mislabel moving history.

    Live bootstrap captures distinct actual idle frames; repeated current images are
    a stationary-scene proxy, not a claim that recorded historical inputs were idle.
    Server-relative timestamps and the exact no-input action match its renderer.
    """
    recent = [dict(sample['current'], timestamp_s=i * 0.1, action=BOOTSTRAP_ACTION)
              for i in range(recent_frames)]
    return dict(current=dict(sample['current'], timestamp_s=recent_frames * 0.1),
                recent=recent, middle=[], notes=[])


def final_checkpoint_control(state, control):
    # Include non-save_steps-aligned endings, with optimizer/RNG state and uploads.
    if state.global_step >= state.max_steps:
        control.should_save = True
    return control


def prepared_image(row, width):
    """Decode the independently pre-resized lossless image, never resample."""
    from PIL import Image
    key = 'thumbnail' if width == 320 else 'image'
    image = row[key]
    if isinstance(image, dict):
        image = Image.open(io.BytesIO(image['bytes']))
    if image.mode != 'RGB' or image.size != (width, round(width * 9 / 16)):
        raise ValueError('Expected pinned pre-resized RGB training image')
    return image


class HistoryDataset:
    """Pinned annotation registry supplies verified, recursive causal memory."""
    def __init__(self, rows, config, index, split, bootstrap_prob=0.0, bootstrap_seed=42):
        import sys
        sys.path.insert(0, str(ROOT / 'legacy'))
        from reader import NormalizedReader
        self.reader = NormalizedReader(index)
        self.ids = self.reader.ids(split)
        self.rows, self.config = rows, config
        self.bootstrap_prob, self.bootstrap_seed = bootstrap_prob, bootstrap_seed
        meta = rows.remove_columns(['image', 'thumbnail', 'action'])
        self.lookup = {(r['session'], r['frame_index']): i for i, r in enumerate(meta)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        raw = self.reader.row(self.ids[i])
        target = raw['source_frames'][raw['target_id']]
        frames = sorted(raw['frames'], key=lambda f: f['timestamp_ms'])
        before = [f for f in frames if f['timestamp_ms'] < target['timestamp_ms']]
        recent = before[-self.config['recent_frames']:]
        cutoff = recent[0]['timestamp_ms'] if recent else target['timestamp_ms']
        middle, last = [], None
        for f in before:
            t = f['timestamp_ms']
            if t >= cutoff or t < target['timestamp_ms'] - 60000:
                continue
            if last is None or t - last >= 5000:
                middle.append(f)
                last = t
        def image_row(f):
            meta = raw['source_frames'][f['id']]
            row = self.rows[self.lookup[(meta['clip'], meta['frame_index'])]]
            # Immutable index binds the exact source action, not just its row position.
            import hashlib
            if hashlib.sha256(row['action'].encode()).hexdigest() != meta['action_sha256']:
                raise ValueError('Source action differs from verified annotation index')
            return row
        sample = {'current': image_row(next(f for f in frames if f['id'] == raw['target_id'])),
                'recent': [image_row(f) for f in recent],
                'middle': [image_row(f) for f in middle], 'notes': raw['notes']}
        # Stable across workers, restarts and subset ordering; validation stays clean.
        if random.Random(f'{self.bootstrap_seed}:{self.ids[i]}').random() < self.bootstrap_prob:
            return bootstrap_sample(sample, self.config['recent_frames'])
        return sample


class ActionCollator:
    def __init__(self, processor, config):
        self.processor, self.config = processor, config

    def messages(self, sample):
        c = self.config
        messages = [{'role': 'system', 'content': SYSTEM}]
        # Reader validates complete source lineage before render-only windowing.
        notes = [f"[{n['timestamp_ms']}ms] {n['text']}" for n in sample.get('notes', [])]
        tokenize = lambda text: self.processor.tokenizer.encode(text, add_special_tokens=False)
        while len(notes) > 1 and len(tokenize('\n'.join(notes))) > c['note_tokens']:
            notes.pop(0)
        if notes:
            text = '\n'.join(notes)
            if len(tokenize(text)) > c['note_tokens']:
                ids = tokenize(text)[:c['note_tokens'] - 16]
                text = self.processor.tokenizer.decode(ids) + ' [memory excerpt]'
            messages.append({'role': 'user', 'content': 'Memory observations (unreviewed):\n' + text})
        for tier in ('middle', 'recent', 'current'):
            rows = [sample['current']] if tier == 'current' else sample[tier]
            for row in rows:
                width = c['thumbnail_width'] if tier == 'middle' else c['frame_width']
                messages.append({'role': 'user', 'content': [
                    {'type': 'text', 'text': f"[{tier} {row['timestamp_s']:.3f}s]"},
                    {'type': 'image', 'image': prepared_image(row, width)}]})
                if tier != 'current':
                    # Past actions are observations, NOT supervised assistant turns.
                    messages.append({'role': 'user', 'content': f"Previous action: {row['action']}"})
        messages.append({'role': 'assistant', 'content': sample['current']['action']})
        return messages

    def encode(self, sample):
        return self.processor.apply_chat_template(
            self.messages(sample), tokenize=True, return_dict=True,
            return_tensors='pt', return_assistant_tokens_mask=True,
            add_generation_prompt=False, enable_thinking=False)

    def __call__(self, samples):
        import torch
        if len(samples) != 1:
            raise ValueError('Multimodal batches use microbatch=1; accumulate gradients instead')
        # Budget the real processor-expanded token sequence, not Qwen constants.
        sample = {k: list(v) if k != 'current' else v for k, v in samples[0].items()}
        # This baseline never changes during history trimming; process it once.
        current = self.encode({'current': sample['current'], 'recent': [], 'middle': []})
        while True:
            batch = self.encode(sample)
            if batch['input_ids'].numel() - current['input_ids'].numel() <= self.config['history_tokens']:
                break
            if sample['middle']:
                sample['middle'].pop(0)
            elif sample['recent']:
                sample['recent'].pop(0)
            elif sample.get('notes'):
                sample['notes'].pop(0)
            else:
                raise ValueError('Cannot satisfy history token budget')
        masks = torch.as_tensor(batch.pop('assistant_masks'), dtype=torch.bool)
        if masks.ndim == 1:
            masks = masks.unsqueeze(0)
        labels = batch['input_ids'].clone()
        labels[~masks] = -100
        # Native generation block includes turn-ending tokens, not role headers.
        if not masks.any():
            raise ValueError('Empty assistant mask: refusing prompt-loss training')
        batch['labels'] = labels
        return batch


def write_license(output, config):
    output = Path(output)
    shutil.copy(ROOT / 'LICENSE-LFM', output / 'LICENSE')
    shutil.copy(ROOT / 'NOTICE', output / 'NOTICE')
    (output / 'README.md').write_text(
        '---\nlicense: other\nlicense_name: lfm-open-license-v1.0\n'
        'license_link: LICENSE\nbase_model: ' + config['model'] + '\n---\n'
        '# ' + config['final_model_repo'] + '\n\n'
        'LFM3-VL-2B derivative fine-tuned on genshinPlayData2 for plain-text actions.\n'
        'Modified weights and chat template; see NOTICE. Training code is MIT, weights are not.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--output', default='/content/lumine-output')
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--subset', type=int, default=0, help='Random training targets; 0 keeps all')
    parser.add_argument('--subset-seed', type=int, default=42)
    parser.add_argument('--bootstrap-prob', type=float, default=0.1,
                        help='Fraction of synthetic idle-start training targets; 0 disables')
    parser.add_argument('--resume')
    parser.add_argument('--action-special-tokens', action='store_true',
                        help='NEXT FRESH run only: add eight action tokens with old-subword mean init')
    parser.add_argument('--keep-remote', type=int, default=2)
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error('--max-steps must be positive, or -1 for the configured epochs')
    if args.subset < 0 or not 0 <= args.bootstrap_prob <= 1:
        parser.error('--subset must be nonnegative and --bootstrap-prob must be in [0, 1]')
    c = json.loads(Path(args.config).read_text())
    # Persist data selection/augmentation in every checkpoint's resume provenance.
    if args.subset:
        c.update(subset=args.subset, subset_seed=args.subset_seed)
    if args.bootstrap_prob:
        c.update(bootstrap_prob=args.bootstrap_prob, bootstrap_seed=args.subset_seed)
    if c.get('action_special_tokens') and not args.action_special_tokens:
        parser.error('Tokenized policy requires --action-special-tokens; this path supports fresh runs only')
    if args.action_special_tokens:
        if args.resume:
            parser.error('--action-special-tokens is for a fresh run, never an old checkpoint')
        c['action_special_tokens'] = True
    import torch
    from datasets import load_dataset, Image as DatasetImage
    from transformers import AutoProcessor, AutoModelForImageTextToText, TrainerCallback, Trainer, TrainingArguments
    from huggingface_hub import HfApi
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Full training requires a BF16-capable GPU; use CPU tests for validation')
    processor = AutoProcessor.from_pretrained(c['model'], revision=c['model_revision'], trust_remote_code=True)
    processor.chat_template = (ROOT / 'chat_template.jinja').read_text()
    model = AutoModelForImageTextToText.from_pretrained(
        c['model'], revision=c['model_revision'], trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation='sdpa')
    action_token_audit = None
    if args.action_special_tokens:
        from action_tokens import initialize_action_tokens
        action_token_audit = initialize_action_tokens(processor, model)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    from prepare import prepare_registry
    index = prepare_registry(Path(args.dataset_dir), c)
    data = load_dataset('parquet', data_files={s: str(Path(args.dataset_dir) / 'data' / (s + '-*.parquet'))
                                             for s in ('train', 'validation')})
    # Decode only the tier actually consumed, not both full and thumbnail images.
    data = data.cast_column('image', DatasetImage(decode=False))
    data = data.cast_column('thumbnail', DatasetImage(decode=False))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run = output.name
    policy = output / 'training-policy.json'
    if policy.exists() and json.loads(policy.read_text()) != c:
        raise ValueError('Resume policy differs from original run')
    policy.write_text(json.dumps(c, indent=2))

    from checkpoint_io import CheckpointUploads, restore
    uploads = CheckpointUploads(output, f"{c['bucket']}/{run}", HfApi(), keep_remote=args.keep_remote)
    if args.resume:
        args.resume = restore(args.resume, output, HfApi())

    class BucketCheckpoint(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            checkpoint = output / f'checkpoint-{state.global_step}'
            processor.save_pretrained(checkpoint)
            if action_token_audit is not None:
                from action_tokens import save_action_tokens
                save_action_tokens(checkpoint, action_token_audit)
            shutil.copy(policy, checkpoint / policy.name)
            write_license(checkpoint, c)
            uploads.submit(checkpoint)

    # Select target IDs, not source rows: historical frame lookup stays intact.
    validation = HistoryDataset(data['validation'], c, index, 'validation')
    validation.ids = [validation.ids[i] for i in sorted(
        random.Random(42).sample(range(len(validation)), min(256, len(validation))))]
    print(f'Evaluation: fixed {len(validation)} targets, every 500 optimizer steps', flush=True)

    training = select_targets(HistoryDataset(data['train'], c, index, 'train',
        bootstrap_prob=args.bootstrap_prob, bootstrap_seed=args.subset_seed), args.subset, args.subset_seed)
    effective_batch = c['gradient_accumulation_steps'] * int(os.environ.get('WORLD_SIZE', '1'))
    steps_per_epoch = math.ceil(math.ceil(len(training) / int(os.environ.get('WORLD_SIZE', '1'))) /
                                c['gradient_accumulation_steps'])
    print(f'Training: {len(training)} targets; effective batch {effective_batch}; '
          f'{steps_per_epoch} optimizer steps/epoch; max_steps={args.max_steps}; '
          f'epoch equivalents={args.max_steps / steps_per_epoch if args.max_steps > 0 else c["epochs"]:.3f}',
          flush=True)

    class EvaluationSchedule(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            return final_checkpoint_control(state, control)

        def on_train_begin(self, training_args, state, control, **kwargs):
            # Trainer restores the old cadence from trainer_state.json on resume.
            state.eval_steps = 500

    trainer = Trainer(model=model, processing_class=processor,
        train_dataset=training,
        eval_dataset=validation,
        data_collator=ActionCollator(processor, c), callbacks=[BucketCheckpoint(), EvaluationSchedule()],
        args=TrainingArguments(output_dir=str(output), bf16=True, fp16=False,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=c['gradient_accumulation_steps'],
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            learning_rate=c['learning_rate'], num_train_epochs=c['epochs'], max_steps=args.max_steps,
            save_steps=c['save_steps'], save_total_limit=None, save_only_model=False,
            eval_strategy='steps', eval_steps=500, prediction_loss_only=True, logging_steps=10,
            # ActionCollator already constructs the assistant-only labels.
            remove_unused_columns=False,
            dataloader_num_workers=args.workers,
            dataloader_persistent_workers=args.workers > 0,
            dataloader_prefetch_factor=2 if args.workers > 0 else None,
            dataloader_multiprocessing_context='spawn' if args.workers > 0 else None,
            dataloader_pin_memory=True, seed=42, data_seed=42, report_to='none'))
    if args.resume:
        saved = Path(args.resume) / 'training-policy.json'
        if not saved.exists() or json.loads(saved.read_text()) != c:
            raise ValueError('Checkpoint provenance does not match training config')
    try:
        trainer.train(resume_from_checkpoint=args.resume)
    finally:
        uploads.close()
    final = output / 'final'
    trainer.save_model(str(final))
    processor.save_pretrained(final)
    if action_token_audit is not None:
        from action_tokens import save_action_tokens
        save_action_tokens(final, action_token_audit)
    shutil.copy(policy, final / policy.name)
    write_license(final, c)
    HfApi().sync_bucket(str(final), f"{c['bucket']}/{run}/final", delete=False)
    # Publishing to final_model_repo remains an explicit, separate user decision.


if __name__ == '__main__':
    main()
