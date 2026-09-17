#!/bin/bash
# Sourced by jobs: compute-node credentials include the newly granted biro group.
set -euo pipefail
umask 027
DAG_ROOT=${DAG_BIRO_ROOT:-/work/hdd/biro/yurenh2/dagformer-20260917}
DAG_SOURCE=${DAG_CODE_SOURCE:-/work/hdd/bfqt/yurenh2/dagformer-eval-20260917}
DAG_REVISION=${DAG_CODE_REVISION:?Pass the committed source revision}
DAG_CODE=$DAG_ROOT/code/$DAG_REVISION
DAG_PYTHON=/u/yurenh2/miniforge3/bin/python3
mkdir -p "$DAG_ROOT/code" "$DAG_ROOT/provenance" "$DAG_ROOT/runs"
(
    flock 9
    if [ ! -f "$DAG_CODE/.git" ]; then
        git --git-dir="$DAG_ROOT/repository.git" fetch "$DAG_SOURCE" "$DAG_REVISION"
        git --git-dir="$DAG_ROOT/repository.git" worktree add --detach "$DAG_CODE" "$DAG_REVISION"
    fi
) 9>"$DAG_ROOT/.code.lock"
cd "$DAG_CODE"
export DAG_ROOT DAG_CODE
export PYTHONPATH="$DAG_CODE"
export HF_HOME="$DAG_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PIP_CACHE_DIR="$DAG_ROOT/cache/pip"
unset TRANSFORMERS_CACHE
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
printf 'Job %s on %s, code %s\n' "$SLURM_JOB_ID" "$(hostname)" "$DAG_REVISION"
