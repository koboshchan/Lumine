"""Local, exact-tokenizer adapter. This never loads model weights or accesses a GPU."""
import os
import json
from functools import lru_cache
from pathlib import Path



@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    root = Path(os.environ["LUMINE_PROCESSOR_DIR"]).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("saved local processor directory required")
    config = json.loads((root / "processor_config.json").read_text())
    vision = config.get("image_processor", {})
    if (config.get("processor_class") != "Qwen3VLProcessor" or
            vision.get("patch_size") != 16 or vision.get("merge_size") != 2 or
            vision.get("temporal_patch_size") != 2):
        raise ValueError("checkpoint processor incompatible with fixed 920/60 image grids")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("saved native chat template required")
    for token in ("<|action_start|>", "<|action_end|>", "<|image_pad|>"):
        if len(tokenizer.encode(token, add_special_tokens=False)) != 1:
            raise ValueError("checkpoint tokenizer missing required intact tokens")
    return tokenizer


def build_step(*, state, frame, executed, now, protocol, max_input_tokens,
               allow_compaction, memory_snapshot=None):
    from .memory import action_step
    tokenizer = _tokenizer()
    return action_step(state=state, frame=frame, executed=executed, now=now,
                       protocol=protocol, max_input_tokens=max_input_tokens,
                       memory_snapshot=memory_snapshot, tokenizer=tokenizer)


async def compact(*, state, now, summarize):
    from .memory import compact_state
    return await compact_state(state=state, now=now, summarize=summarize,
                               tokenizer=_tokenizer())


build_step.compact = compact
build_step.snapshot_mode = True


async def close():
    from .memory import shutdown
    shutdown()


build_step.close = close
