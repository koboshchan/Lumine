# Experimental native vLLM serving

Status on November 21, 2026: exact CPU preprocessing optimization passed the
256-sample A100 gate below. Native vLLM processor caching FAILED and stays off.
The existing Transformers server and all training defaults remain unchanged.
The pinned vLLM reference and optimized path produced identical outputs in the
November 21 gate; this is not an arbitrary-model or runtime parity guarantee.

Use a separate environment with `pip install -r training/lfm3/requirements-vllm.txt`.
vLLM 0.34.0 requires Torch 2.14.0, torchvision 0.29.0, Pillow >=13 and
Transformers >=5.16.1,<5.23. Never install it over the training environment.
Only load trusted checkpoint directories; their processor executes Python code.

```sh
export LUMINE_SERVE_KEY='<your private key>'
python training/lfm3/serve_vllm.py --checkpoint /path/to/final
```

The listener defaults to localhost. Use an encrypted private tunnel. Authentication,
session reset/bootstrap/step, ACK/replay handling and action validation are the same
as `serve.py`. No server or controller starts on import. This does not change the
runner transport. It accepts the existing 640x360 pre-resized lossless PNG route;
that skips the ingress downsize. Keep the client's training-parity LANCZOS resize.
GPU interpolation is not substituted for LANCZOS.

The backend requests native single-GPU BF16, CUDA graphs, prefix caching disabled, greedy
sampling, a 96-token maximum, and an included `<|action_end|>` stop delimiter.
The public engine handles bounded multimodal processor and GPU encoder caches,
using server-computed RGB content/resolution hashes. Processor caching is currently disabled after the failure below. Processor caching is not
itself evidence of GPU embedding reuse. GPU encoder entries can be evicted;
"each frame once" is not an unbounded lifetime guarantee. Native encoder-call
metrics must verify reuse on the target runtime.

Prompt ordering is exactly the trained chronological ordering. History remains
before the current frame, but sliding windows, tier labels, timestamps and notes
can invalidate KV prefixes. The budget uses actual native-expanded image tokens
and evicts oldest middle/recent/notes in the existing order. Full image token
parity must pass before scoring. No FP8 is enabled or claimed faster on A100.
`--eager` and `--no-prefix-cache` expose diagnostic ablations.

## Reproducible GPU gate

`benchmark_vllm.py` runs one backend per process, 256 fixed seed-42 held-out targets,
and checks the exact previous selection SHA256
`10d4f2811c5012cbf8e7acaf545a651d890bc54f4b7fd2b7bead3d77dacc6bb8`.
Dataset revision is `daff7e95b133fcb1c5f9f8504c7be4c2e952e902`.
Use a root containing `data/data/validation*.parquet`, verified
`data/expanded.sqlite`, `lumine-lfm3-colab-test-1h/` checkpoint files and the
previous eval's `prior.jsonl`. The model is the Nov17 run's `final` directory in
`hf://buckets/koboshchan/checkpoints/lumine-lfm3-colab-test-1h/final`.

```sh
python training/lfm3/benchmark_vllm.py --root /path/to/eval --backend transformers --output results/transformers.jsonl
python training/lfm3/benchmark_vllm.py --root /path/to/eval --backend vllm-eager --output results/vllm-eager.jsonl
python training/lfm3/benchmark_vllm.py --root /path/to/eval --backend vllm-graphs --output results/vllm-graphs.jsonl
python training/lfm3/benchmark_vllm.py --root /path/to/eval --backend vllm-cached --output results/vllm-cached.jsonl
```

Raw records preserve the cold first request; p50/p95 omit it. Reports include
strict string exact match, parser validity, Transformers agreement, effective
history count, and preprocessing/encode/prefill/decode/generation/total times.
Transformers components use CUDA events; vLLM components use engine timestamps
and synchronized encoder timing. These component timings are not kernel-equivalent.
Full wall-clock generation/total are the primary end-to-end comparison.
The cache ablation compares eager/no-cache, graphs/no-cache and graphs/caches;
it does not isolate prefix and image caches from each other.

## Actual November 20 results

29 CPU tests passed with Torch 2.14.0+cpu, torchvision 0.29.2+cpu,
Transformers 5.22.1 and Pillow 13.0.0. The native vLLM processor source and trained
HF image processor produced identical normalized pixels for one real held-out
640x360 image: max/mean absolute difference 0, matching grid [1,22,40].
This is preliminary CPU evidence, not a 256-sample GPU parity result.
Pre-resized frames avoid a new resize, so their resize pixel delta is zero by
construction; full-resolution LANCZOS GPU parity was not implemented or measured.

A100 80GB, driver 580.82.07, BF16, same fixed 256 targets and exact checkpoint.
All three valid runs parsed 256/256; effective recent history was 4 through 8.
Each pair below is p50 / p95 seconds, excluding the first cold request.

Transformers exact 78/256 (30.46875%). Preprocess 0.543 / 2.002;
encode 0.052 / 0.052; prefill 0.068 / 0.087; decode 0.695 / 0.868;
generation wall 0.880 / 1.075; total wall 1.435 / 2.955.

vLLM eager, caches off, exact 80/256 (31.25%), agreement 245/256 (95.703%).
Preprocess 0.841 / 2.698; encode 0.039 / 0.045; prefill 0.078 / 0.083;
decode 0.491 / 0.630; generation wall 0.736 / 0.887; total 1.546 / 3.462.

vLLM graphs, caches off, exact 80/256 (31.25%), agreement 247/256 (96.484%).
Preprocess 0.802 / 2.633; encode 0.047 / 0.053; prefill 0.070 / 0.081;
decode 0.077 / 0.097; generation wall 0.334 / 0.364; total 1.133 / 2.985.
Median total improved about 21%, but p95 did not improve. Generation-only
improved about 2.63x. Components have different timing definitions above;
medians are not additive. This is not bit-identical output parity.
The first four complete sampled prompts passed native/HF expanded-token and
normalized-pixel equality, max/mean delta zero. This is not all-256 pixel auditing.

The graphs plus processor/prefix cache ablation FAILED all 256 requests with
native P0/P1 multimodal cache-drift errors, empty output and no encoder calls.
Its nominal 0.561s median total is an ERROR response time, not inference latency.
Its prefill timestamps are invalid. Do not quote these as a speedup or embedding
reuse. Preparing prompts with the renderer can advance its frontend cache without
advancing the engine cache. Processor caching is now explicitly rejected until a
corrected lifecycle passes the GPU gate. Prefix caching alone and encoder reuse
were not isolated or validated; both caches default off; prefix caching is available only via explicit
`--prefix-cache`, an unmeasured experimental combination. No deployment occurred.
The benchmark cache ablation intentionally fails closed with that safety guard.

35 local tests pass. All allocated runtimes were unassigned; final live balance
84.70 CU, zero assignments and zero usage rate. Starting pre-retry balance was
90.99 CU (6.29 CU used), original pre-task balance 91.11 CU. Several setup and
instrumentation failures preceded the measured run; none are claimed as results.
No FP8 or GPU LANCZOS substitution was enabled. Further GPU work needs approval.

## November 21 exact preprocessing optimization

Exact fast preprocessing is now the default; use `--no-fast-prep` for the old
path. The HTTP API, training, prompt chronology, greedy sampling, BF16 and
PIL LANCZOS ingress resize are unchanged. Prefix caching and native multimodal
processor caching remain off. No GPU interpolation, quantization, native P0/P1
cache, or GPU embedding cache was introduced.

On the same A100 80GB and fixed 256 held-out targets, a fresh back-to-back
reference/optimized comparison produced identical expanded prompt IDs and
identical completions on all 256. Both scored 80/256 exact ground-truth matches
(31.25%), parsed 256/256, with effective recent history 4 through 8. This is
100% agreement with the current vLLM reference, not with Transformers.

Times below are p50 / p95 seconds, excluding the cold first request. This is
in-process inference, not capture-to-action or HTTP network latency.

Reference history preparation was 0.800 / 2.585; optimized 0.172 / 1.475.
Reference generation was 0.335 / 0.364; optimized 0.350 / 0.389.
Reference total was 1.135 / 2.948; optimized 0.548 / 1.825.
Total median improved 51.7% (2.07x) and p95 improved 38.1%. Generation is not
faster; the win is avoiding redundant preparation. Component medians do not add.
Encoder was 0.0467 / 0.0537 versus 0.0469 / 0.0534; prefill 0.0697 / 0.0814
versus 0.0696 / 0.0719; decode 0.0773 / 0.0982 versus 0.0746 / 0.0940.

CPU profiling found image decoding/hashing and full HF processing repeated for
baseline/full/trim history-budget probes. The fast path uses the official pinned
processor's shape-to-patch-count rule for token budgeting, a 128-frame decoded
PNG LRU, and a separate 256 MiB CPU pixel-tensor LRU. The latter stores the exact
stock HF resize/normalize/patchify result, keyed by processor instance, RGB
content, dimensions and processing kwargs. Its outputs are fresh concatenations,
so a caller cannot mutate cached entries. Engine access is serialized. History
trimming order remains middle, recent, notes. New processor versions/checkpoints
must re-run the full expanded-token parity gate before using this path.

Measured native CPU resize stage p50 / p95 fell 0.0800 / 0.1175 to
0.0168 / 0.0209 seconds; rescale/normalize 0.0321 / 0.0486 to
0.0070 / 0.0093; patchify 0.0516 / 0.0765 to 0.0118 / 0.0139.
Decode-cache hits/misses were 950/4033, pixel-cache hits/misses 767/3688,
with 254.7 MiB retained. These randomly selected trajectories do not imply one
lifetime preprocessing call per frame, and misses are expected after eviction.
There were still 256 native encoder calls: this does not cache GPU embeddings.

The CPU-only 256-prompt parity pass measured budget p50 / p95 of
0.492 / 1.591 versus 0.095 / 0.782 seconds. Normalized pixels from the exact
cache versus stock mixed-size batching had max/mean absolute difference zero;
the GPU benchmark's first four full prompt pixel audits also had zero delta.
All 256 expanded prompt IDs and generated outputs were compared on GPU, but
all-256 normalized-pixel tensors were not archived or separately audited.

A representative local 1280x720 PNG measured p50 decode 11.69 ms, base64 decode
0.45 ms, crop 0.12 ms, PIL LANCZOS 640x360 resize 12.00 ms and 320x180 resize
9.13 ms. These are CPU microbenchmarks, not A100 timings; PNG compression/content
and CPU speed matter. JPEG is rejected by the current HTTP API. Client-side
640x360 lossless PNG already skips ingress downsize, so no protocol change was
needed. HTTP/base64 is outside the offline dataset benchmark. Actual network
transfer and isolated CPU/GPU copy time were not measured; generation wall time
includes any transfers. No pinned-memory benefit or GPU-resize equivalence is
claimed. Exact cached stock transforms were preferred over approximate GPU
interpolation, since repeated history work was the larger measured opportunity.

Reproduce with `profile_preprocessing.py --root /path/to/eval --output cpu.json`,
then run `benchmark_vllm.py` with `--backend vllm-reference` and
`--backend vllm-fast`, writing both JSONL outputs to the same directory. The
reference uses `transformers.jsonl` from the earlier gate as its comparison;
the fast run instead compares against `vllm-reference.jsonl` and asserts every
expanded prompt ID. Committed summaries are in `benchmarks/prep-nov21/`.
The backend-local cache patch depends on pinned vLLM 0.34.0's processor interface.

38 CPU tests and compilation passed. The single A100 session was released after
about 22 minutes, with independently verified zero active sessions/assignments
and 0.00 units/hour. Balance was 80.40 units afterwards; the balance delta also
includes an unrelated session already active before this one, so it is not an
isolated cost measurement. No serving/controller process or watcher was left.
