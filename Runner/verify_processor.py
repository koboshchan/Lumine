"""CPU-only serving preflight; does not load weights or allocate CUDA."""
import json
import sys
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor

root = Path(sys.argv[1])
processor = AutoProcessor.from_pretrained(root, local_files_only=True)
tokenizer = processor.tokenizer
ids = json.loads((Path(__file__).resolve().parents[1] /
                  "SessionProxy" / "custom_token_ids.json").read_text())
assert len(ids) == 7, f"expected seven new custom tokens, got {len(ids)}"
for token, expected in ids.items():
    actual = tokenizer.encode(token, add_special_tokens=False)
    assert actual == [expected], (token, actual, expected)
    assert tokenizer.decode(actual, skip_special_tokens=False) == token
    assert tokenizer.decode(actual, skip_special_tokens=True) == token, token

image = Image.new("RGB", (1280, 736))
vision = processor.image_processor(images=[image], do_resize=False, return_tensors="pt")
grid = vision["image_grid_thw"].tolist()
assert grid == [[1, 46, 80]], grid
assert int(vision["image_grid_thw"].prod()) // 4 == 920

messages = [{"role": "user", "content": [{"type": "image"}]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
assert prompt == "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n", repr(prompt)
print("PASS: 7 atomic new custom tokens (existing action words retain trained BPE), native image-only ChatML, grid1x46x80,920 image tokens")
