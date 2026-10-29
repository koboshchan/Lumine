#!/bin/bash
# Run from any directory. Data paths resolve relative to this Lumine checkout.
# Usage: ./train.sh [0|1|2|3|all] [--gpu <positive integer>]
# VEOMNI_DIR may select an updated fork checkout; otherwise use the submodule
# when initialized, or a sibling ../VeOmni checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STAGE="all"
GPU_COUNT=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        0|1|2|3|all)
            STAGE="$1"
            shift
            ;;
        --gpu)
            if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                echo "--gpu requires a positive integer" >&2
                exit 1
            fi
            GPU_COUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${VEOMNI_DIR:-}" ]]; then
    if [[ -f "$SCRIPT_DIR/VeOmni/tasks/train_vlm.py" ]]; then
        VEOMNI_DIR="$SCRIPT_DIR/VeOmni"
    else
        VEOMNI_DIR="$SCRIPT_DIR/../VeOmni"
    fi
fi
if [[ ! -f "$VEOMNI_DIR/tasks/train_vlm.py" ]]; then
    echo "Set VEOMNI_DIR to an updated VeOmni fork with tasks/train_vlm.py" >&2
    exit 1
fi
VEOMNI_DIR="$(cd "$VEOMNI_DIR" && pwd)"

# Use the selected checkout's environment when available.
if [[ -f "$VEOMNI_DIR/.venv/bin/activate" ]]; then
    source "$VEOMNI_DIR/.venv/bin/activate"
fi
export PYTHONPATH="$VEOMNI_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

run_stage() {
    local stage="$1"
    local config="$2"
    local log_file="$SCRIPT_DIR/stage${stage}_training.log"

    echo "Running stage $stage on $GPU_COUNT GPU(s)"
    echo "Config: $config"
    echo "Log file: $log_file"
    if torchrun --standalone --nproc-per-node="$GPU_COUNT" \
        "$VEOMNI_DIR/tasks/train_vlm.py" "$config" 2>&1 | tee "$log_file"; then
        echo "Stage $stage completed successfully!"
    else
        echo "Stage $stage failed!" >&2
        return 1
    fi
}

STAGE_NAMES=(knowledge pretrain instruct reasoning)
if [[ "$STAGE" == "all" ]]; then
    for stage in 0 1 2 3; do
        run_stage "$stage" "$SCRIPT_DIR/configs/stage${stage}_${STAGE_NAMES[$stage]}.yaml"
    done
    echo "All training stages completed!"
else
    run_stage "$STAGE" "$SCRIPT_DIR/configs/stage${STAGE}_${STAGE_NAMES[$STAGE]}.yaml"
fi
