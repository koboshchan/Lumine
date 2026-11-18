# Lumine

A Genshin Impact vision-to-action agent, fine-tuning [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B) toward **koboshchan/Lumine-Mini-2B**. It learns to predict mouse and keyboard inputs from gameplay frames and causal history.

An independent implementation inspired by [Lumine: An Open Recipe for Building Generalist Agents in 3D Open Worlds](https://arxiv.org/abs/2511.08892) (Tan et al., 2025), not the paper's official code or a reproduction of its reported results.

[Training](training/lfm3/README.md) · [VM setup & resume](training/lfm3/VM_SETUP.md) · [Serving](training/lfm3/SERVING.md) · [Windows runner](Runner/README.md) · [Dataset](https://huggingface.co/datasets/koboshchan/genshinPlayData2)

## Current recipe

The active training code is in [training/lfm3](training/lfm3). It uses full-parameter supervised fine-tuning, with loss only on the current action. Previous actions are observations, not additional prediction targets.

| Component | Configuration |
| --- | --- |
| Base model | `LiquidAI/LFM3-VL-2B`, pinned revision |
| Data | `koboshchan/genshinPlayData2`, pinned revision and verified shard hashes |
| Images | Independent RGB 640×360 frames and 320×180 thumbnails |
| History | Eight recent frames, older thumbnails sampled every five seconds within a 60-second window, and causal memory notes |
| History budget | 4,096 tokens, excluding the current frame and target action |
| Training defaults | One epoch, learning rate `2e-5`, microbatch 1, gradient accumulation 8 |
| Action vocabulary | 15 special tokens with `--action-special-tokens` |

Exact revisions, hashes and policy settings live in [config.json](training/lfm3/config.json). The older Qwen/VeOmni configs and scripts are legacy paths, not this recipe.

> [!IMPORTANT]
> Training also requires the private, hash-pinned annotation registry under `lumine/immutable-payload-v1` in the HF bucket `koboshchan/checkpoints`. The public dataset alone is not enough to reproduce the run. There are no gameplay benchmarks or paper-level results claimed here.

## Train

Use the [Colab notebook](notebooks/lumine-mini-2b-lfm3.ipynb) for a short test, or follow the [VM setup guide](training/lfm3/VM_SETUP.md) for environment, driver and storage preparation. Training requires a BF16-capable CUDA GPU; the VM guide documents its untested GPU assumptions.

Set `HF_TOKEN` privately with model/dataset read access and private bucket read/write access. In Colab, use the `HF_TOKEN` secret. After installing the pinned [requirements](training/lfm3/requirements.txt), download the pinned data from the repository root:

```bash
python - <<'PY'
import json
from huggingface_hub import snapshot_download

c = json.load(open("training/lfm3/config.json"))
snapshot_download(
    c["dataset"], repo_type="dataset", revision=c["dataset_revision"],
    local_dir="./lumine-data",
    allow_patterns=["data/*.parquet", "resize-manifest.json"],
)
PY

python -u training/lfm3/train.py \
  --dataset-dir ./lumine-data \
  --output ./runs/lfm3-action15-test-001 \
  --subset 1000 --max-steps 1000 --subset-seed 42 \
  --action-special-tokens --bootstrap-prob 0.1 --keep-remote 2
```

Use a unique output directory for each fresh run. On one GPU, this 1,000-target test repeats the subset for eight epochs; it is not a full-data run. Remove `--subset` and `--max-steps` for the configured full epoch.

Fresh action-token runs initialize the 15 token embeddings from their original subword means. By default, 10% of training samples use synthetic idle-start history: eight copies of the current image paired with no-input actions, with older history and notes cleared. The target stays unchanged. Set `--bootstrap-prob 0` to disable it. Live bootstrap instead requires actual chronological idle captures, not fabricated history.

### Checkpoints and resume

The trainer saves full state every 100 optimizer steps and at the final step, including optimizer, scheduler, RNG, processor, tokenizer and policy. Checkpoint tar archives upload asynchronously to `hf://buckets/koboshchan/checkpoints/<output-basename>`. Each upload is checked by remote size and downloaded-byte SHA-256 before retention keeps the newest two archives. Failed uploads retain their local files.

Action-token resume is supported from [72270b5](https://github.com/koboshchan/Lumine/commit/72270b5). Add one of these options to the **original training command**, keeping the same output basename and original flags:

```bash
# Highest complete checkpoint in the local output directory
--resume latest

# Highest checkpoint archive in the original private bucket run
--resume "hf://buckets/koboshchan/checkpoints/lfm3-action15-test-001/latest"
```

Keep the same source, dependencies, GPU count, data selection, bootstrap and training schedule. Resume validates saved token IDs and vocabulary, loads the resized embeddings without reinitializing them, and restores training state. It does not silently fall back to a fresh run. See the [resume guide](training/lfm3/VM_SETUP.md#spot-preemption-and-full-state-resume) for explicit checkpoint paths and recovery details.

Final model artifacts upload to the same private bucket. Publishing weights to `koboshchan/Lumine-Mini-2B` is a separate step, not an automatic training side effect.

## Run a checkpoint

Use [training/lfm3/serve.py](training/lfm3/serve.py) for LFM3 checkpoints, with the saved tokenizer, action-token audit and policy intact. See the [serving guide](training/lfm3/SERVING.md) for setup, but use the current 15-token policy described above rather than its older eight-token notes or the legacy Qwen/vLLM runner examples.

The authenticated `/v1/session/step` path maintains history from acknowledged, executed actions; `reset` clears it and `bootstrap` seeds an idle session. The OpenAI-compatible `/v1/chat/completions` route is single-frame debugging, not history-aware gameplay.

The native [Windows runner](Runner/README.md) captures frames and parses actions. Enable **Session proxy mode** for the session transport. Dry run is the default; F8 stops execution and releases held inputs. Start by inspecting predictions without enabling input.

> [!WARNING]
> Keep serving on loopback behind an encrypted tunnel and set `LUMINE_SERVE_KEY`. Screenshots and credentials are sensitive. Load only trusted checkpoints, since model custom code and serialized training state can execute code. Native capture and in-game input still need live validation; no anti-cheat bypass is included.

## Recording and data tools

[OBSInputLogger](OBSInputLogger/README.md) records input sidecars during explicitly enabled OBS recordings on Windows. Input capture is system-wide while recording, so close private apps and record only with consent. Align the sidecar with its video before passing it to [DataProcessor](DataProcessor); see the [action format](DataProcessor/action_format.md) for the mouse prefix and six key chunks per action. [KeyRecorder](KeyRecorder) is the older standalone recorder.

Code is [MIT](LICENSE). Model weights and the modified upstream chat template remain under the [LFM Open License v1.0](training/lfm3/LICENSE-LFM), not MIT. Preserve the upstream license and [notices](training/lfm3/NOTICE) when distributing derivatives.
