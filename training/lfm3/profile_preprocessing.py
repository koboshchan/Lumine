"""CPU-only profiling of exact ingress, history budgeting and pixel transforms.

Run against the pinned offline-eval root, never a live controller or HTTP server.
HTTP network time, video capture, and GPU transfer are not CPU measurements.
"""
import argparse
import base64
import io
import json
import random
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=256)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from datasets import load_dataset, Image as DatasetImage
    from train import HistoryDataset
    from serve import load_checkpoint_processor
    from serve_vllm import PromptBuilder
    from frame_cache import PixelFrames
    torch.set_num_threads(4)
    p, config = load_checkpoint_processor(args.root / 'lumine-lfm3-colab-test-1h')
    data = load_dataset('parquet', data_files=str(args.root / 'data/data/validation*.parquet'))['train']
    data = data.cast_column('image', DatasetImage(decode=False)).cast_column('thumbnail', DatasetImage(decode=False))
    ds = HistoryDataset(data, config, args.root / 'data/expanded.sqlite', 'validation')
    ds.ids = [ds.ids[i] for i in sorted(random.Random(42).sample(range(len(ds)), 256))]
    def render(raw):
        batch = p(text=raw['prompt'], images=raw['multi_modal_data']['image'],
                  add_special_tokens=False, return_tensors='pt')
        return {'prompt_token_ids': batch['input_ids'][0].tolist()}
    old = PromptBuilder(p, config, render)
    fast = PromptBuilder(p, config, render, fast=True)
    records = []
    for i in range(args.samples):
        sample = ds[i]
        start = time.perf_counter()
        raw, expected, _ = old.prepare(sample)
        old_time = time.perf_counter() - start
        start = time.perf_counter()
        new_raw, actual, _ = fast.prepare(sample)
        new_time = time.perf_counter() - start
        assert actual['prompt_token_ids'] == expected['prompt_token_ids']
        records.append(dict(i=i, budget_old_s=old_time, budget_fast_s=new_time))
    # Stock batch processor versus cached per-frame processing, exact equality.
    images = new_raw['multi_modal_data']['image']
    original = p.image_processor(images, return_tensors='pt')
    cache = PixelFrames()
    cache.install(p.image_processor)
    cached = p.image_processor(images, return_tensors='pt')
    diff = (original['pixel_values'] - cached['pixel_values']).abs()
    assert torch.equal(original['image_grid_thw'], cached['image_grid_thw'])
    assert diff.max() == 0
    assert torch.equal(p.image_processor(images, return_tensors='pt')['pixel_values'], original['pixel_values'])
    stages = {}
    image = images[-1].resize((1280, 720), Image.Resampling.NEAREST)
    def measure(name, fn):
        times = []
        for _ in range(12):
            start = time.perf_counter()
            value = fn()
            times.append(time.perf_counter() - start)
        stages[name] = {'p50': statistics.median(times[1:]), 'p95': sorted(times[1:])[-1]}
        return value
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    png = buffer.getvalue()
    encoded = base64.b64encode(png)
    measure('base64_decode', lambda: base64.b64decode(encoded, validate=True))
    def decode():
        im = Image.open(io.BytesIO(png)); im.load()
        return im
    measure('png_decode_1280', decode)
    measure('crop_1280', lambda: image.crop((0, 0, 1280, 720)))
    measure('pil_lanczos_640', lambda: image.resize((640, 360), Image.Resampling.LANCZOS))
    measure('pil_lanczos_320', lambda: image.resize((320, 180), Image.Resampling.LANCZOS))
    measure('pixel_cache_hit_all_frames', lambda: p.image_processor(images, return_tensors='pt'))
    result = {'n': len(records), 'records': records, 'stages_s': stages,
              'pixel_delta_max': diff.max().item(), 'pixel_delta_mean': diff.mean().item(),
              'scope': 'CPU only; no HTTP network/capture or GPU transfer time. Dataset already resized PNG; JPEG rejected by HTTP API.'}
    for name in ('budget_old_s', 'budget_fast_s'):
        values = sorted(r[name] for r in records[1:])
        result[name] = {'p50': statistics.median(values), 'p95': values[max(0, int(len(values)*.95)-1)]}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
