"""Offline, sequential latency/accuracy comparison on the fixed Nov20 holdout.

No HTTP listener, input controller, training, allocation or background watcher.
Run each backend in a fresh process; never keep both models in GPU memory.
Preprocessing parity is checked before scoring, including post-trim histories.
"""
import argparse
import ast
import re
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

SELECTION_SHA = '10d4f2811c5012cbf8e7acaf545a651d890bc54f4b7fd2b7bead3d77dacc6bb8'


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)]


def summarize(records):
    result = {'n': len(records), 'selection_sha256': SELECTION_SHA,
              'exact_ground_truth': sum(r['generated'] == r['truth'] for r in records) / len(records),
              'agreement_transformers': sum(r['generated'] == r['baseline'] for r in records) / len(records),
              'parse_valid': sum(r['parse_error'] is None for r in records)}
    # First request is cold. Raw records preserve it separately.
    for name in ('preprocess_s', 'encode_s', 'prefill_s', 'decode_s', 'generate_s', 'total_s'):
        values = [r[name] for r in records[1:] if r.get(name) is not None]
        result[name] = {'p50': statistics.median(values), 'p95': percentile(values, .95)} if values else None
    result['effective_recent_range'] = [min(r['recent_effective'] for r in records),
                                        max(r['recent_effective'] for r in records)]
    result['encoder_calls'] = sum(r.get('encoder_calls', 0) for r in records)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--backend', choices=['transformers', 'vllm-eager', 'vllm-graphs', 'vllm-cached'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    from datasets import load_dataset, Image as DatasetImage
    from serve import Backend, encode_prompt
    from serve_vllm import VLLMBackend
    from train import HistoryDataset
    # Reuse the exact project parser without importing its service or controllers.
    repo = Path(__file__).resolve().parents[2]
    source = ast.parse((repo / 'SessionProxy/service.py').read_text())
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_completion']
    namespace = {'re': re, 'MAX_COMPLETION': 32 * 1024, 'KEYS': set(
        'Esc Tab Caps Shift Ctrl Alt Space Back Enter Left Up Right Down LB RB MB '
        'zero one two three four five six seven eight nine '
        'One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve'.split()) | set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'project-parser', 'exec'), namespace)
    parse_action = namespace['parse_completion']
    torch.set_num_threads(4)
    torch.manual_seed(42)
    checkpoint = args.root / 'lumine-lfm3-colab-test-1h'
    data = load_dataset('parquet', data_files={'validation': str(args.root / 'data/data/validation*.parquet')})['validation']
    data = data.cast_column('image', DatasetImage(decode=False)).cast_column('thumbnail', DatasetImage(decode=False))
    policy = json.loads((checkpoint / 'training-policy.json').read_text())
    dataset = HistoryDataset(data, policy, args.root / 'data/expanded.sqlite', 'validation')
    dataset.ids = [dataset.ids[i] for i in sorted(random.Random(42).sample(range(len(dataset)), 256))]
    assert hashlib.sha256(json.dumps(dataset.ids).encode()).hexdigest() == SELECTION_SHA
    prior = [json.loads(line) for line in (args.root / 'prior.jsonl').read_text().splitlines()]
    assert [r['id'] for r in prior] == dataset.ids
    baseline_path = args.output.parent / 'transformers.jsonl'
    baseline = ([json.loads(line) for line in baseline_path.read_text().splitlines()]
                if args.backend != 'transformers' else prior)
    assert len(baseline) == 256
    cached = args.backend == 'vllm-cached'
    backend = (Backend(checkpoint) if args.backend == 'transformers' else
               VLLMBackend(checkpoint, eager=args.backend == 'vllm-eager',
                           prefix_cache=cached, mm_cache_gb=1 if cached else 0, timing=True))
    events, handles = [], []
    def hook_before(module, inputs):
        start = torch.cuda.Event(enable_timing=True); start.record()
        events.append([module._bench_name, start, None])
    def hook_after(module, inputs, output):
        end = torch.cuda.Event(enable_timing=True); end.record()
        next(item for item in reversed(events) if item[0] == module._bench_name and item[2] is None)[2] = end
    if args.backend == 'transformers':
        # Forward events are GPU elapsed, excluding Python/scheduler gaps.
        for name, module in [('model', backend.model), ('vision', backend.model.model.vision_tower),
                             ('projector', backend.model.model.multi_modal_projector)]:
            module._bench_name = name
            handles.extend([module.register_forward_pre_hook(hook_before), module.register_forward_hook(hook_after)])
    records = []
    pixel_report = []
    with args.output.open('w') as file:
        for i in range(256):
            sample = dataset[i]
            truth = sample['current']['action']
            assert truth == prior[i]['truth']
            events.clear()
            torch.cuda.synchronize(); started = time.perf_counter()
            if args.backend == 'transformers':
                with torch.inference_mode():
                    batch = encode_prompt(backend.processor, policy, sample).to('cuda')
                    torch.cuda.synchronize(); prepared = time.perf_counter()
                    prompt_ids = batch['input_ids'][0].tolist()
                    output = backend.model.generate(**batch, max_new_tokens=96, do_sample=False, use_cache=True)
                    torch.cuda.synchronize(); finished = time.perf_counter()
                    ids = output[0, len(prompt_ids):].tolist()
                    eos = backend.model.generation_config.eos_token_id
                    end = set(eos if isinstance(eos, list) else [eos]); end.add(backend.processor.tokenizer.pad_token_id)
                    while ids and ids[-1] in end: ids.pop()
                    generated = backend.processor.tokenizer.decode(ids, skip_special_tokens=False)
                forwards = [a.elapsed_time(b)/1000 for name, a, b in events if name == 'model']
                encode = sum(a.elapsed_time(b)/1000 for name, a, b in events if name in ('vision', 'projector'))
                timing = dict(preprocess_s=prepared-started, encode_s=encode,
                              prefill_s=max(0, forwards[0]-encode), decode_s=sum(forwards[1:]),
                              generate_s=finished-prepared, total_s=finished-started)
                effective = backend.processor.tokenizer.decode(prompt_ids, skip_special_tokens=False).count('[recent ')
                calls = sum(name == 'vision' for name, *_ in events)
                del batch, output
            else:
                # Gate native processor/token count parity on held-out actual frames.
                # This is deliberately outside measured latency and only first 4.
                if i < 4:
                    raw, rendered, eff = backend.builder.prepare(sample)
                    reference = encode_prompt(backend.processor, policy, sample)
                    assert rendered['prompt_token_ids'] == reference['input_ids'][0].tolist(), 'native prompt drift'
                    from vllm.transformers_utils.processors.lfm3_vl import Lfm3VlProcessor
                    native = Lfm3VlProcessor.from_pretrained(checkpoint, trust_remote_code=True)
                    images = raw['multi_modal_data']['image']
                    x = backend.processor.image_processor(images=images, return_tensors='pt')
                    y = native.image_processor(images=images, return_tensors='pt')
                    diff = (x['pixel_values'] - y['pixel_values']).abs()
                    pixel_report.append({'i': i, 'max_abs_normalized': diff.max().item(), 'mean_abs_normalized': diff.mean().item()})
                    assert diff.max() == 0, 'native pixels differ'
                torch.cuda.synchronize(); started = time.perf_counter()
                generated, _, _ = backend.complete(sample, 96)
                torch.cuda.synchronize(); finished = time.perf_counter()
                stats = backend.engine.collective_rpc('get_encoder_timing_stats')
                entries = [entry for worker in stats for entry in worker.values()]
                encode = sum(entry['encoder_forward_secs'] for entry in entries)
                calls = sum(entry['num_encoder_calls'] for entry in entries)
                m = backend.last_output.metrics
                prefill = max(0, m.first_token_ts - m.scheduled_ts - encode) if m else None
                decode = max(0, m.last_token_ts - m.first_token_ts) if m else None
                timing = dict(backend.last_timings, encode_s=encode, prefill_s=prefill,
                              decode_s=decode, total_s=finished-started)
                # No extra processor invocation in the measured request.
                prompt_ids = backend.last_output.prompt_token_ids
                effective = backend.processor.tokenizer.decode(prompt_ids, skip_special_tokens=False).count('[recent ')
            try:
                parse_action(generated)
                parse_error = None
            except ValueError as exc:
                parse_error = str(exc)
            record = dict(parse_error=parse_error, i=i, id=dataset.ids[i], truth=truth, generated=generated,
                          baseline=baseline[i]['generated'], recent_effective=effective,
                          prompt_tokens=len(prompt_ids), encoder_calls=calls, **timing)
            records.append(record); file.write(json.dumps(record)+'\n'); file.flush()
            if i % 16 == 0: print(args.backend, i+1, timing, flush=True)
    result = summarize(records)
    result['pixel_parity'] = pixel_report
    result['timing_definition'] = ('TF encode/prefill/decode are CUDA-event forward time; generation/total wall time. '
        'vLLM encode synchronized native encoder wall, prefill scheduler-to-first-token minus encode; '
        'decode first-to-last-token wall. Cross-engine component timings are not kernel-equivalent.')
    result['gpu'] = torch.cuda.get_device_name()
    result['peak_vram_gib'] = (torch.cuda.max_memory_allocated()/1024**3
                               if args.backend == 'transformers' else None)
    args.output.with_suffix('.summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
