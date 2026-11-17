# Fresh Ubuntu 24.04 x86_64 H100 VM

These commands are for a **fresh single-H100 VM**, not Colab. The CPU tests and
Python 3.12 Linux dependency resolution were checked; driver installation,
full-model GPU backward, H100 memory fit, and this VM procedure have not been run.
Do not apply the driver upgrade to an occupied VM. Stop any image-provided
automatic driver installer first; do not mix a runfile-installed driver with APT.
Secure Boot requires signed/enrolled kernel modules or an appropriate image.

NVIDIA's [CUDA 13.2 GA release notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/)
list **595.45.04** as its Linux toolkit driver version. The separate CUDA 13.x
minor-compatibility floor is 580, with restrictions including newer PTX.
CUDA 13.2 Update 2 lists **595.71.05** in its
[release notes](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html).
The resolved wheel runtime is 13.2.86, so the commands conservatively require
R595 at least 595.71.05 rather than relying on that restricted floor.
PyTorch 2.15.0's cu132 wheels ship their CUDA runtime; installing a system CUDA
toolkit is unnecessary. There is no cu129 wheel for the pinned torch/torchvision.

The [official NVIDIA Ubuntu driver procedure](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html)
uses cuda-keyring, a branch-pinning package, and the unversioned nvidia-open
package for branches 590 and later. Current Ubuntu 24.04 NVIDIA repository metadata
includes nvidia-driver-pinning-595 and nvidia-open 595.112.07.

Run before reboot, reviewing APT's proposed changes. Existing package holds/pins
or an image-provided installer can prevent switching branches; resolve those
explicitly rather than purging arbitrary packages.

```bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential dkms linux-headers-"$(uname -r)" \
  ca-certificates curl git python3.12-venv tmux
cd /tmp
curl -fSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install nvidia-driver-pinning-595
apt-cache policy nvidia-open
sudo apt-get install nvidia-open
sudo reboot
```

Reconnect after reboot.

```bash
set -euo pipefail
nvidia-smi
while read -r driver; do
  dpkg --compare-versions "$driver" ge 595.71.05 || {
    echo "Driver too old: $driver"; exit 1;
  }
done < <(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
```

Prepare **one blank bundled local NVMe SSD** using its stable Google by-id name,
never a guessed /dev/nvme0n1 (which could be the boot disk). Check the inventory
first. The formatting step below refuses any existing signature, partition,
mount, or holder. If the image already mounted/formatted its SSD, use that
existing mount instead and set the paths below accordingly; do not reformat it.
This script formats only a confirmed blank local SSD, destroying anything
unrecognized on that device. Local SSD is ephemeral. Remote checkpoints are
essential. One SSD is used here; this is not a multi-disk RAID setup.

```bash
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
ls -l /dev/disk/by-id/google-local-nvme-ssd-*
(
  set -euo pipefail
  DEV=/dev/disk/by-id/google-local-nvme-ssd-0
  REAL=$(readlink -f "$DEV")
  test -b "$REAL"
  test "$(lsblk -dnro TYPE "$REAL")" = disk
  test "$(lsblk -nrpo NAME "$REAL" | wc -l)" -eq 1
  test -z "$(lsblk -nrpo MOUNTPOINTS "$REAL" | tr -d '[:space:]')"
  test -z "$(ls "/sys/class/block/${REAL##*/}/holders")"
  test -z "$(sudo wipefs -n --noheadings "$REAL")"
  test -z "$(find /mnt/disks/lumine -mindepth 1 -print -quit 2>/dev/null || true)"
  sudo mkfs.ext4 "$DEV"
  sudo mkdir -p /mnt/disks/lumine
  sudo mount -o discard,defaults "$DEV" /mnt/disks/lumine
  sudo chown "$(id -u):$(id -g)" /mnt/disks/lumine
)
findmnt /mnt/disks/lumine
df -h /mnt/disks/lumine
```

Allow ample space for the dataset, Arrow cache, model, optimizer checkpoints,
upload tar files and SHA read-back copies. Uploads queue asynchronously, so peak
usage can exceed one local checkpoint. Remount the SSD after a subsequent reboot
before using these paths; no persistent mount is installed here.

```bash
set -euo pipefail
export WORK=/mnt/disks/lumine
export HF_HOME="$WORK/hf"
export TMPDIR="$WORK/tmp"
mkdir -p "$HF_HOME" "$TMPDIR" "$WORK/runs"
# Enter a token with access to the dataset/model and write access to the private bucket.
read -rsp 'HF token: ' HF_TOKEN; echo
export HF_TOKEN

git clone https://github.com/koboshchan/Lumine.git "$WORK/Lumine"
cd "$WORK/Lumine"
git rev-parse HEAD | tee "$WORK/source-commit.txt"
python3.12 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
python -m pip install --upgrade pip
# The venv does not inherit image packages. Remove audio if present; it is unused
# and an incompatible torchaudio binary can break transformers imports.
python -m pip uninstall -y torchaudio
python -m pip install torch==2.15.0 torchvision==0.29.2 \
  --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r training/lfm3/requirements.txt
python -m pip install pytest fastapi httpx aiohttp
python -m pip check
python - <<'PY'
import torch, torchvision
print(torch.__version__, torchvision.__version__, torch.version.cuda)
assert torch.__version__.split('+')[0] == '2.15.0'
assert torchvision.__version__.split('+')[0] == '0.29.2'
assert torch.version.cuda == '13.2'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(torch.cuda.get_device_name(0))
PY
# CPU-only acceptance, native processor and tiny random decoders, no GPU training.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest training/lfm3/tests -v
python - <<'PY'
import json
from huggingface_hub import snapshot_download
c = json.load(open('training/lfm3/config.json'))
snapshot_download(c['dataset'], repo_type='dataset', revision=c['dataset_revision'],
                  local_dir='/mnt/disks/lumine/data',
                  allow_patterns=['data/*.parquet', 'resize-manifest.json'])
PY
# The trainer verifies all shard hashes and the private immutable annotation index
# before training. This audit reads only action columns, not full model weights.
CUDA_VISIBLE_DEVICES='' python training/lfm3/audit_action_tokens.py \
  --output "$WORK/action-token-audit.json"
```

Full run is one epoch of **all** targets, not the 1,000-target smoke test.
The config pins model 23387beda43c60310348e83e25f78f33c3140d3a and dataset
daff7e95b133fcb1c5f9f8504c7be4c2e952e902, learning rate 2e-5, accumulation 8,
save every 100 optimizer steps. On one GPU, microbatch 1 gives effective batch 8.
Bootstrap probability is 0.1, validation is a fixed 256-target sample every 500
steps, and final-step optimizer checkpoint saving is forced even off cadence.
The private destination is hf://buckets/koboshchan/checkpoints/<run>, newest two
verified remote checkpoint archives retained, plus final artifacts. No public
model push happens. Use a unique output basename for every fresh run.

```bash
export RUN="lfm3-full-action15-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export RUN_DIR="$WORK/runs/$RUN"
mkdir "$RUN_DIR"
cat > "$RUN_DIR/train.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disks/lumine/Lumine
source /mnt/disks/lumine/venv/bin/activate
export OMP_NUM_THREADS=8
python -u training/lfm3/train.py \
  --dataset-dir /mnt/disks/lumine/data \
  --output "$RUN_DIR" \
  --workers 4 \
  --action-special-tokens \
  --bootstrap-prob 0.1 \
  --keep-remote 2 2>&1 | tee "$RUN_DIR/train.log"
SH
chmod 700 "$RUN_DIR/train.sh"
# A separate tmux socket ensures the NEW server inherits HF_TOKEN and path vars.
tmux -L "$RUN" new-session -d -s train "$RUN_DIR/train.sh"
echo "Session socket: $RUN"
echo "Log file: $RUN_DIR/train.log"
tail -f "$RUN_DIR/train.log"
# Ctrl-C stops tail, not the detached trainer.
# To attach: tmux -L "$RUN" attach -t train
```

## Spot preemption and full-state resume

Action15 checkpoints can now resume with `--action-special-tokens`. The saved
tokenizer must contain the exact same fifteen action tokens, IDs and vocabulary
size (125046), and its audit must agree. Eight-token/legacy or otherwise changed
tokenizers fail before model loading. Resuming loads the saved resized embeddings
directly, without adding tokens, resizing, or mean-initializing them again.
Trainer restores optimizer, scheduler, RNG, global step and skips consumed batches
in the same shuffled epoch. Keep the same source commit, dependencies, config,
GPU count, subset/seed, bootstrap probability, accumulation and max-step/epoch
schedule. New checkpoints also record data-order settings and reject changes.
Older action15 checkpoints without this extra settings file support single-GPU
resume if their existing policy, RNG and scheduler provenance match.

Record the exact `RUN` string somewhere persistent before starting. After a spot
VM restart/replacement, remount a surviving disk, or recreate the environment and
re-download the pinned dataset if the local SSD was lost. Do not reformat a
surviving disk or repeat the fresh-run command that generates a new run name.
Restore `WORK`, `HF_HOME`, `TMPDIR`, `HF_TOKEN`, `RUN` and `RUN_DIR` as above.
The output basename must remain the original run name so uploads stay in its
existing private bucket prefix.

Run this once, manually or from your own VM startup procedure. It discovers the
highest numbered archive in this exact run prefix and fails if none exists;
it never silently starts a fresh run. This is not a VM provisioner or retry loop.

```bash
export WORK=/mnt/disks/lumine
export HF_HOME="$WORK/hf" TMPDIR="$WORK/tmp"
export RUN='lfm3-full-action15-PASTE-ORIGINAL-RUN-NAME'
export RUN_DIR="$WORK/runs/$RUN"
mkdir -p "$RUN_DIR" "$HF_HOME" "$TMPDIR"
cd "$WORK/Lumine"
source "$WORK/venv/bin/activate"
export OMP_NUM_THREADS=8
# HF_TOKEN must be exported again after a reboot.
# If the exact checkpoint directory is already present locally, use
# --resume latest instead of extracting the same remote tar over it.
python -u training/lfm3/train.py \
  --dataset-dir "$WORK/data" \
  --output "$RUN_DIR" \
  --workers 4 \
  --action-special-tokens \
  --bootstrap-prob 0.1 \
  --keep-remote 2 \
  --resume "hf://buckets/koboshchan/checkpoints/$RUN/latest"
```

For an explicitly selected checkpoint, replace the last argument with
`hf://buckets/koboshchan/checkpoints/$RUN/checkpoint-1200/checkpoint-1200.tar`,
a local `/path/to/checkpoint-1200.tar`, or a checkpoint directory.
`--resume latest` means the highest complete **local** checkpoint in
`RUN_DIR`; `hf://…/$RUN/latest` discovers the latest **remote** archive.
An existing destination directory is never overwritten by tar extraction;
use its directory path or local `latest` on a surviving disk.
If the original run used `--subset`, `--subset-seed` or `--max-steps`,
repeat those exact options. Do not set max steps to the number of steps remaining.
A completed checkpoint should not be restarted as a new run.

Resume only checkpoints you own and trust. Processor/model custom Python and
Trainer RNG/optimizer state are executable or serialized training artifacts,
not a safe format for untrusted uploads. Tar restoration rejects traversal,
links and non-checkpoint roots. Only fully uploaded archives are discoverable,
but interrupted/corrupt downloads still fail rather than falling back to a
different checkpoint. Up to one save interval plus upload lag can be lost on
preemption; a checkpoint sitting only on ephemeral storage is not durable.
