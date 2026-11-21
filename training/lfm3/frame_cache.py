"""Bounded, exact CPU caches. No native vLLM P0/P1 cache or GPU embeddings."""
import hashlib
import io
import json
import time
from collections import OrderedDict

from PIL import Image


class DecodedFrames:
    """Accept only immutable encoded bytes or server-owned immutable RGB images."""
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.items = OrderedDict()
        self.seconds = 0.0
        self.hits = self.misses = 0

    def get(self, value):
        if not isinstance(value, dict):
            return value
        encoded = value.get('bytes')
        if not isinstance(encoded, bytes):
            raise ValueError('Expected inline encoded frame bytes')
        key = hashlib.sha256(encoded).digest()
        if key in self.items:
            self.hits += 1
            self.items.move_to_end(key)
            return self.items[key]
        started = time.perf_counter()
        image = Image.open(io.BytesIO(encoded))
        if image.mode != 'RGB' or image.size not in ((640, 360), (320, 180)):
            raise ValueError('Expected pinned RGB frame or thumbnail')
        image.load()
        self.seconds += time.perf_counter() - started
        self.misses += 1
        self.items[key] = image
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return image

    def sample(self, sample):
        def row(value, key):
            value = dict(value)
            if key in value:
                value[key] = self.get(value[key])
            return value
        return dict(sample, current=row(sample['current'], 'image'),
                    recent=[row(r, 'image') for r in sample['recent']],
                    middle=[row(r, 'thumbnail') for r in sample['middle']])


class PixelFrames:
    """Memoize stock HF resize/normalize/patchify, never approximate pixels.

    Entries include processor kwargs and immutable RGB content, with a byte cap.
    Unsupported input/output forms bypass caching. Access is serialized by the
    backend lock, including native renderer calls. No client cache IDs are used.
    """
    def __init__(self, max_bytes=256 * 1024**2):
        self.max_bytes = max_bytes
        self.items = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0
        self.seconds = 0.0

    def install(self, processor):
        if getattr(processor, '_lumine_pixel_cache', None) is self:
            return
        original = processor.preprocess
        def preprocess(images, **kwargs):
            import torch
            from transformers.feature_extraction_utils import BatchFeature
            values = [images] if isinstance(images, Image.Image) else images
            if (not isinstance(values, (list, tuple)) or not values
                    or not all(isinstance(im, Image.Image) and im.mode == 'RGB' for im in values)
                    or kwargs.get('return_tensors') not in ('pt', None)):
                return original(images, **kwargs)
            try:
                signature = json.dumps(kwargs, sort_keys=True)
            except (TypeError, ValueError):
                return original(images, **kwargs)
            batches = []
            for image in values:
                digest = hashlib.sha256(image.tobytes()).digest()
                key = (id(processor), image.size, digest, signature)
                if key in self.items:
                    self.hits += 1
                    self.items.move_to_end(key)
                    batch, _ = self.items[key]
                else:
                    self.misses += 1
                    started = time.perf_counter()
                    batch = original([image], **dict(kwargs, return_tensors='pt'))
                    self.seconds += time.perf_counter() - started
                    if set(batch) != {'pixel_values', 'image_grid_thw'}:
                        return original(images, **kwargs)
                    size = sum(t.numel() * t.element_size() for t in batch.values())
                    if size <= self.max_bytes:
                        self.items[key] = (batch, size)
                        self.bytes += size
                        while self.bytes > self.max_bytes:
                            _, (_, removed) = self.items.popitem(last=False)
                            self.bytes -= removed
                batches.append(batch)
            # Concatenation creates fresh tensors, so callers cannot mutate entries.
            return BatchFeature({name: torch.cat([b[name] for b in batches], dim=0)
                                 for name in ('pixel_values', 'image_grid_thw')},
                                tensor_type=kwargs.get('return_tensors'))
        processor.preprocess = preprocess
        processor._lumine_pixel_cache = self
