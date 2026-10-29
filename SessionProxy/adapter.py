"""Local, exact-tokenizer adapter. This never loads model weights or accesses a GPU."""
import os
import base64
import io
import json
from functools import lru_cache
from pathlib import Path

from lumine_memory.live import build_live_step


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


def build_step(**kwargs):
    tokenizer = _tokenizer()
    if kwargs["protocol"] == "tiered-v1":
        from PIL import Image
        raw = base64.b64decode(kwargs["frame"].split(",", 1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            small = image.convert("RGB").crop((0, 0, 1280, 720)).resize((320, 180), Image.Resampling.LANCZOS)
            padded = Image.new("RGB", (320, 192), "black")
            padded.paste(small, (0, 0))
            output = io.BytesIO()
            padded.save(output, format="PNG")
            kwargs["thumbnail"] = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    return build_live_step(**kwargs, apply_template=tokenizer.apply_chat_template,
                           encode=lambda s: tokenizer.encode(s, add_special_tokens=False))
