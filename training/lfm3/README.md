# LFM3 full-parameter action SFT

Training code remains MIT. Fine-tuned weights and the modified upstream template use the accompanying LFM Open License v1.0. `NOTICE` identifies modifications. Destination is configured once in `config.json` as `koboshchan/Lumine-Mini-2B`; final artifacts upload to the private Lumine checkpoints bucket, not directly to that model repository.

The minimal Colab notebook fetches main and captures its exact commit for the run, installs pinned dependencies, reads `HF_TOKEN` from Colab Secrets, downloads the pinned pre-resized dataset and its hash-bound resize manifest, and runs `train.py`. It creates no GPU runtime and contains no watchdog/controller. Choose a BF16 GPU yourself. GPU memory fit and throughput have not been measured.

The entrypoint loads the exact LFM3 processor and custom model code at the pinned revision. The shipped native template defaults thinking off and removes empty thinking blocks while preserving generation tags. Only the final assistant action is supervised via the processor's assistant mask; history actions are user observations. No tool schemas, LoRA, frozen vision encoder, or Qwen image-token constants. Fresh runs can enable the eight action tokens with `--action-special-tokens`; their embeddings use the old subword means and every checkpoint saves the processor/tokenizer and token audit. This is not an upgrade path for the old checkpoint-1000 tokenizer.

Frames and independent thumbnails are decoded at 640x360 and 320x180 without training-time resizing. The resized manifest binds every shard to the immutable full-resolution source hashes; full-resolution data stays available in dataset history.

Causal memory comes from the existing hash-pinned annotation index in the private Lumine bucket. The ported reader verifies source-note recursion, timestamps, clip/split boundaries, evidence references, and source-action hashes. History uses up to eight recent 640-wide frames, 320-wide middle thumbnails every five seconds over a minute, and at most 1024 tokens of unreviewed teacher memory. The actual LFM3 processor-expanded history is bounded to 4096 tokens, evicting oldest middle/recent observations when needed. Current frame and target are outside the history budget. Exact tokenizer-length budgeting differs from the former Qwen processor, intentionally.

Full optimizer/scheduler/RNG checkpoints save every 100 optimizer steps, upload asynchronously as one uncompressed tar per checkpoint with remote size/SHA256 read-back; training continues during upload. They upload to `hf://buckets/koboshchan/checkpoints/<run>/checkpoint-N`, and retain only the newest verified local checkpoint and the two newest remote archives. Final weights, processor, license and model card upload to the same run's `final` directory. Use a unique output directory per run. For resume, download a complete checkpoint and pass `--resume /path/to/checkpoint-N`; config mismatch is rejected. Publishing model weights remains a separate decision.

Run CPU acceptance with `python -m pytest training/lfm3/tests -v`. Tests use native processor rendering/masks/image grids and a tiny randomly initialized LFM3 text decoder. They do not establish full-sized multimodal GPU backward, memory fit, training quality, or bucket-upload performance.

Acceptance: native template, image/assistant mask and two tiny native decoder backward/optimizer steps passed on CPU with torch 2.15.0+cpu and torchvision 0.29.2+cpu. The causal-reader rejection test also passed. Full GPU training was not run.


## Short fresh Colab test

```bash
python -u training/lfm3/train.py --dataset-dir /content/lumine-data --output /content/lumine-lfm3-test-1k --workers 8 --max-steps 1000 --subset 1000 --subset-seed 42 --action-special-tokens --bootstrap-prob 0.1
```

`--max-steps` defaults to -1 (configured epoch count); `--subset` defaults to 0 (all targets). The seed defaults to 42. Sampling chooses random training target IDs, not source rows: each selected target retains its complete original causal image/action/note context. Validation remains the same seed-42 sample of 256 targets, with no augmentation. Asking for more targets than exist is an error.

On one GPU, microbatch 1 times gradient accumulation 8 gives effective batch 8. A 1,000-target subset has 125 optimizer steps per epoch, so 1,000 steps means **8 epochs**, not one pass. This is a repeatability/overfit smoke test, not a new independent 8,000-example dataset. On the previously measured A100 80GB throughput of roughly 6.4–6.8 seconds per step, allow roughly 1.8–1.9 hours for optimizer steps alone, plus preparation, eval and uploads; shorter/synthetic histories can change throughput. No real training is launched by the tests.

Training-only `--bootstrap-prob` defaults to 0.1; set 0 to disable. The selected fraction is deterministic per target and seed across workers/resumes. Synthetic startup examples clear old notes and middle history, repeat the current stationary scene as eight recent frames, and assign the server's exact no-input action and relative capture timestamps. Live bootstrap uses distinct *actual* idle captures; repetition is only a training proxy, not fabricated labels on moving historical frames. The current target action is unchanged and remains the only supervised turn. Existing non-augmented samples preserve all context. This augmentation has acceptance tests, not evidence that it fixes live gameplay.

Data selection and augmentation settings are included in checkpoint policy provenance. Use the same flags on resume. To resume older full-run checkpoints whose policy predates augmentation, pass `--bootstrap-prob 0` and do not enable new action tokens. The final optimizer step always saves a complete `checkpoint-N`, even if N is not divisible by 100, and waits for its verified upload before final artifacts. The processor/tokenizer and policy are saved in both checkpoints and final artifacts.
