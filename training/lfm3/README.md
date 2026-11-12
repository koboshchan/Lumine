# LFM3 full-parameter action SFT

Training code remains MIT. Fine-tuned weights and the modified upstream template use the accompanying LFM Open License v1.0. `NOTICE` identifies modifications. Destination is configured once in `config.json` as `koboshchan/Lumine-Mini-2B`; final artifacts upload to the private Lumine checkpoints bucket, not directly to that model repository.

The minimal Colab notebook fetches main and captures its exact commit for the run, installs pinned dependencies, reads `HF_TOKEN` from Colab Secrets, downloads the pinned pre-resized dataset and its hash-bound resize manifest, and runs `train.py`. It creates no GPU runtime and contains no watchdog/controller. Choose a BF16 GPU yourself. GPU memory fit and throughput have not been measured.

The entrypoint loads the exact LFM3 processor and custom model code at the pinned revision. The shipped native template defaults thinking off and removes empty thinking blocks while preserving generation tags. Only the final assistant action is supervised via the processor's assistant mask; history actions are user observations. No tool schemas, added action tokens, LoRA, frozen vision encoder, or Qwen image-token constants.

Frames and independent thumbnails are decoded at 640x360 and 320x180 without training-time resizing. The resized manifest binds every shard to the immutable full-resolution source hashes; full-resolution data stays available in dataset history.

Causal memory comes from the existing hash-pinned annotation index in the private Lumine bucket. The ported reader verifies source-note recursion, timestamps, clip/split boundaries, evidence references, and source-action hashes. History uses up to eight recent 640-wide frames, 320-wide middle thumbnails every five seconds over a minute, and at most 1024 tokens of unreviewed teacher memory. The actual LFM3 processor-expanded history is bounded to 4096 tokens, evicting oldest middle/recent observations when needed. Current frame and target are outside the history budget. Exact tokenizer-length budgeting differs from the former Qwen processor, intentionally.

Full optimizer/scheduler/RNG checkpoints save every 100 optimizer steps, upload synchronously to `hf://buckets/koboshchan/checkpoints/<run>/checkpoint-N`, and retain the last two locally. Final weights, processor, license and model card upload to the same run's `final` directory. Use a unique output directory per run. For resume, download a complete checkpoint and pass `--resume /path/to/checkpoint-N`; config mismatch is rejected. Publishing model weights remains a separate decision.

Run CPU acceptance with `python -m pytest training/lfm3/tests -v`. Tests use native processor rendering/masks/image grids and a tiny randomly initialized LFM3 text decoder. They do not establish full-sized multimodal GPU backward, memory fit, training quality, or bucket-upload performance.

Acceptance: native template, image/assistant mask and two tiny native decoder backward/optimizer steps passed on CPU with torch 2.15.0+cpu and torchvision 0.29.2+cpu. The causal-reader rejection test also passed. Full GPU training was not run.
