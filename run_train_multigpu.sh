#!/bin/bash
# Usage: bash run_train_multigpu.sh runs/tacit/config.yaml [NUM_GPUS]
# The paper models were trained on 4 GPUs (batch size 16 per GPU, effective batch size 64).
set -euo pipefail

CONFIG=${1:-runs/tacit/config.yaml}
NGPUS=${2:-4}

cd "$(dirname "$0")"

torchrun --standalone --nproc_per_node="${NGPUS}" src/bond_match/train_ranking.py --config "${CONFIG}"
