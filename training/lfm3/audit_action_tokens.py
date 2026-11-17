"""CPU-only audit of every pinned dataset action, including complete chat rendering.

Reads only the action column via HTTP ranges, never downloads images/model weights.
Run from the repository root with python training/lfm3/audit_action_tokens.py.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq
from transformers import AutoProcessor

from action_tokens import ACTION_TOKENS, register_action_tokens
from train import ActionCollator, SYSTEM

ROOT = Path(__file__).resolve().parent


def read_actions(item):
    dataset, revision, path = item
    fs = HfFileSystem()
    with fs.open(f'datasets/{dataset}@{revision}/{path}', 'rb',
                 block_size=65536, cache_type='none') as handle:
        actions = pq.ParquetFile(handle).read(columns=['action']).column('action').to_pylist()
    if not all(isinstance(action, str) for action in actions):
        raise ValueError(f'Non-string action in {path}')
    return path, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'action-token-audit.json'))
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    config = json.loads((ROOT / 'config.json').read_text())
    processor = AutoProcessor.from_pretrained(config['model'], revision=config['model_revision'],
                                              trust_remote_code=True)
    processor.chat_template = (ROOT / 'chat_template.jinja').read_text()
    tokenizer = processor.tokenizer
    before = len(tokenizer)
    previous = {word: tokenizer.encode(word, add_special_tokens=False) for word in ACTION_TOKENS}
    probes = [SYSTEM, 'Capsule sixteen seventeen ninety Seventeen Eighteen Ninety '
              'Elevenfold Twelvefold LBrace']
    probes += [f'x{word} {word}x _{word} {word}_ é{word} {word}é'
               for word in ACTION_TOKENS if word.isalnum()]
    unchanged = {probe: tokenizer.encode(probe, add_special_tokens=False) for probe in probes}
    register_action_tokens(tokenizer)
    for probe, ids in unchanged.items():
        assert tokenizer.encode(probe, add_special_tokens=False) == ids, probe
    token_ids = {word: tokenizer.convert_tokens_to_ids(word) for word in ACTION_TOKENS}
    for word, token_id in token_ids.items():
        assert tokenizer.encode(word, add_special_tokens=False) == [token_id], word
    paths = sorted(entry.path for entry in HfApi().list_repo_tree(
        config['dataset'], repo_type='dataset', revision=config['dataset_revision'], recursive=True)
        if entry.path.endswith('.parquet'))
    if len(paths) != 94:
        raise ValueError(f'Unexpected shard count: {len(paths)}')
    digest = hashlib.sha256()
    actions = Counter()
    inventory = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for path, rows in pool.map(read_actions, [(config['dataset'], config['dataset_revision'], p)
                                                for p in paths]):
            inventory[path] = len(rows)
            digest.update((json.dumps([path, rows], ensure_ascii=False, separators=(',', ':')) + '\n').encode())
            actions.update(rows)
            print(f'Read {path}: {len(rows)} actions', flush=True)
    # Render the actual full template with system, memory, a past action and a
    # current image+assistant target. Every real action is checked in BOTH roles.
    from PIL import Image
    sentinel = 'LUMINE_ACTION_AUDIT_SENTINEL'
    frame = dict(image=Image.new('RGB', (640, 360)), timestamp_s=1.0, action=sentinel)
    sample = dict(current=frame, recent=[frame], middle=[],
                  notes=[dict(timestamp_ms=100, text=probes[1])])
    rendered = processor.apply_chat_template(ActionCollator(processor, config).messages(sample),
                                            tokenize=False, add_generation_prompt=False,
                                            enable_thinking=False)
    assert rendered.count(sentinel) == 2
    unique = sorted(actions)
    errors = []
    occurrences = Counter()
    raw_count = full_count = 0
    lexical = {word: re.compile(r'(?<!\w)' + re.escape(word) + r'(?!\w)')
               for word in ACTION_TOKENS if word.isalnum()}
    for offset in range(0, len(unique), 256):
        batch = unique[offset:offset + 256]
        prompts = [rendered.replace(sentinel, action) for action in batch]
        for kind, texts in [('action', batch), ('prompt', prompts)]:
            encoded = tokenizer(texts, add_special_tokens=False)['input_ids']
            decoded = tokenizer.batch_decode(encoded, skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)
            for action, text, ids, result in zip(batch, texts, encoded, decoded):
                if result != text:
                    errors.append(dict(kind=kind, action=action, decoded=result))
                counts = Counter(ids)
                for word, pattern in lexical.items():
                    expected = len(pattern.findall(text))
                    if counts[token_ids[word]] != expected:
                        errors.append(dict(kind=kind, action=action, non_atomic=word,
                                           expected=expected, actual=counts[token_ids[word]]))
                    if kind == 'action':
                        occurrences[word] += expected * actions[action]
                if kind == 'action':
                    raw_count += len(ids) * actions[action]
                else:
                    full_count += len(ids) * actions[action]
    report = dict(model=config['model'], model_revision=config['model_revision'],
                  dataset=config['dataset'], dataset_revision=config['dataset_revision'],
                  vocab_before=before, vocab_after=len(tokenizer), action_special_tokens=len(ACTION_TOKENS),
                  tokens={word: dict(id=token_ids[word], previous_ids=previous[word],
                                      single_word=word.isalnum(), lstrip=False, rstrip=False,
                                      normalized=False, special=True) for word in ACTION_TOKENS},
                  shards=inventory, rows=sum(actions.values()), unique_actions=len(actions),
                  ordered_action_column_sha256=digest.hexdigest(),
                  lexical_occurrences=dict(occurrences), system_and_embedded_probes_unchanged=True,
                  full_prompt_roles=['previous user action', 'current assistant action'],
                  action_tokens_total=raw_count, full_prompt_tokens_total=full_count,
                  skip_special_tokens=False, changed_decoded_text=errors)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('tokens', 'shards')}, indent=2))
    if errors:
        raise SystemExit('Audit failed, see changed_decoded_text')


if __name__ == '__main__':
    main()
