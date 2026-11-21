"""Optional single-GPU backend. Keep training in its separately pinned environment.

Run only trusted, locally extracted checkpoints: their HF processor is Python code.
The existing authenticated HTTP/session adapter and action checks are unchanged.
"""
import argparse
import hashlib
import os
import threading
import time
from collections import OrderedDict
from importlib.metadata import version

from packaging.version import Version
from serve import create_app, load_checkpoint_processor
from train import ActionCollator


class FrameHashes:
    """Bounded identity LRU; content-derived IDs never come from HTTP clients.

    Retaining the image prevents Python id reuse. Images are immutable after
    request decoding. Resolution and mode are included, so thumbnails cannot
    alias their full-sized source. The engine owns its bounded processor and
    GPU encoder caches, not this small identity map.
    """
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.items = OrderedDict()

    def get(self, image):
        key = id(image)
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key][1]
        h = hashlib.sha256()
        h.update(f'{image.mode}:{image.width}:{image.height}:'.encode())
        h.update(image.tobytes())
        value = h.hexdigest()
        self.items[key] = (image, value)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return value


class PromptBuilder:
    """Budget actual native-processor tokens without re-encoding old pixels.

    Do not reorder the trained prompt for cache hits. Notes, middle and recent
    history stay before current, byte-for-byte the training template. Sliding
    windows and tier/timestamp changes can invalidate KV prefix hits. Identical
    frames still reuse native multimodal processor/encoder entries until eviction.
    """
    def __init__(self, processor, config, render):
        self.processor, self.config, self.render = processor, config, render
        self.hashes = FrameHashes()

    def raw(self, sample):
        messages = ActionCollator(self.processor, self.config).messages(sample)[:-1]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        images = [part['image'] for message in messages
                  if isinstance(message['content'], list)
                  for part in message['content'] if part.get('type') == 'image']
        return {'prompt': text, 'multi_modal_data': {'image': images},
                'multi_modal_uuids': {'image': [self.hashes.get(im) for im in images]}}

    def prepare(self, sample):
        sample = dict(sample, recent=list(sample['recent']), middle=list(sample['middle']),
                      notes=list(sample.get('notes', [])))
        baseline = self.render(self.raw(dict(sample, recent=[], middle=[], notes=[])))
        baseline_tokens = len(baseline['prompt_token_ids'])
        while True:
            raw = self.raw(sample)
            rendered = self.render(raw)
            if len(rendered['prompt_token_ids']) - baseline_tokens <= self.config['history_tokens']:
                return raw, rendered, sample
            for tier in ('middle', 'recent', 'notes'):
                if sample[tier]:
                    sample[tier].pop(0)
                    break
            else:
                raise ValueError('Cannot satisfy history token budget')


class VLLMBackend:
    def __init__(self, checkpoint, *, eager=False, prefix_cache=False,
                 mm_cache_gb=0, timing=False):
        if Version(version('vllm')) < Version('0.34.0'):
            raise RuntimeError('Native LFM3-VL requires vllm>=0.34.0')
        if mm_cache_gb:
            raise ValueError('Processor cache is disabled: native renderer cache drift failed GPU validation')
        from vllm import LLM
        self.processor, self.config = load_checkpoint_processor(checkpoint)
        self.lock = threading.Lock()
        # BF16 on A100. No FP8 path is enabled without a measured speed/accuracy win.
        # max_tokens is capped separately; cache allocations are bounded per process.
        self.engine = LLM(
            model=str(checkpoint), tokenizer=str(checkpoint), trust_remote_code=True,
            dtype='bfloat16', tensor_parallel_size=1, max_num_seqs=1,
            max_model_len=8192, max_num_batched_tokens=8192,
            gpu_memory_utilization=0.8, enforce_eager=eager,
            enable_prefix_caching=prefix_cache,
            mm_processor_cache_gb=mm_cache_gb,
            limit_mm_per_prompt={'image': 32, 'video': 0},
            mm_processor_kwargs={'size': {'longest_edge': 230400, 'shortest_edge': 65536}},
            generation_config='vllm', seed=42,
            enable_mm_processor_stats=timing, disable_log_stats=not timing,
        )
        self.builder = PromptBuilder(self.processor, self.config, self.render)
        self.last_output = None

    def render(self, prompt):
        # Native renderer expands image tokens and caches processing by content ID.
        # Passing a rendered string must not add a second BOS token.
        params = self.engine.renderer.default_cmpl_tok_params.with_kwargs(add_special_tokens=False)
        return self.engine.renderer.render_cmpl([prompt], params)[0]

    def complete(self, sample, max_tokens):
        from vllm import SamplingParams
        if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise ValueError('max_tokens must be an integer in 1..512')
        with self.lock:
            started = time.perf_counter()
            raw, _, _ = self.builder.prepare(sample)
            prepared = time.perf_counter()
            params = SamplingParams(temperature=0, max_tokens=min(max_tokens, 96),
                stop=['<|action_end|>'], include_stop_str_in_output=True,
                skip_special_tokens=False, spaces_between_special_tokens=False)
            # Re-rendering is a processor-cache hit. Use the public generation API
            # rather than submitting private engine inputs across release boundaries.
            output = self.engine.generate([raw], params, use_tqdm=False,
                tokenization_kwargs={'add_special_tokens': False})[0]
            self.last_timings = {'preprocess_s': prepared - started,
                                 'generate_s': time.perf_counter() - prepared}
            self.last_output = output
            completion = output.outputs[0]
            if (completion.finish_reason == 'error' or not completion.text
                    or '<|action_end|>' not in completion.text):
                raise RuntimeError('vLLM failed to return a complete action; no action acknowledged')
            return completion.text, len(output.prompt_token_ids), len(completion.token_ids)


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model-name', default='lumine-mini-2b')
    parser.add_argument('--eager', action='store_true', help='Diagnostic ablation, disable CUDA graphs')
    parser.add_argument('--no-prefix-cache', action='store_true')
    parser.add_argument('--prefix-cache', action='store_true', help='Experimental, not separately GPU-validated')
    args = parser.parse_args()
    key = os.environ.get('LUMINE_SERVE_KEY', '')
    if not key:
        parser.error('Set LUMINE_SERVE_KEY in the environment')
    backend = VLLMBackend(args.checkpoint, eager=args.eager, prefix_cache=args.prefix_cache and not args.no_prefix_cache)
    uvicorn.run(create_app(backend, key, args.model_name), host=args.host, port=args.port, access_log=False)


if __name__ == '__main__':
    main()
