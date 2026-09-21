#!/bin/bash
# Sourced by jobs: compute-node credentials include the newly granted biro group.
set -euo pipefail
umask 027
DAG_ROOT=${DAG_BIRO_ROOT:-/work/hdd/biro/yurenh2/dagformer-20260917}
DAG_HOME=${DAG_HOME_ROOT:-/u/yurenh2/dagformer-20260917}
DAG_SOURCE=${DAG_CODE_SOURCE:-/work/hdd/bfqt/yurenh2/dagformer-eval-20260917}
DAG_REVISION=${DAG_CODE_REVISION:?Pass the committed source revision}
DAG_CODE=$DAG_HOME/code/$DAG_REVISION
DAG_PYTHON=/u/yurenh2/miniforge3/bin/python3
mkdir -p "$DAG_HOME/code" "$DAG_HOME/provenance" "$DAG_ROOT/runs"
(
    flock 9
    if [ ! -f "$DAG_CODE/.git" ]; then
        if [ ! -d "$DAG_HOME/repository.git" ]; then
            git clone --bare "$DAG_SOURCE" "$DAG_HOME/repository.git"
        fi
        git --git-dir="$DAG_HOME/repository.git" fetch "$DAG_SOURCE" "$DAG_REVISION"
        git --git-dir="$DAG_HOME/repository.git" worktree add --detach "$DAG_CODE" "$DAG_REVISION"
    fi
) 9>"$DAG_HOME/.code.lock"
# Keep old submitted scripts and absolute checkpoint/tokenizer paths usable.
# Existing physical directories are moved by the storage migration, not here.
for DAG_SMALL_DIR in code repository.git envs cache results provenance tokenizer; do
    if [ ! -e "$DAG_ROOT/$DAG_SMALL_DIR" ] && [ ! -L "$DAG_ROOT/$DAG_SMALL_DIR" ]; then
        ln -s "$DAG_HOME/$DAG_SMALL_DIR" "$DAG_ROOT/$DAG_SMALL_DIR"
    fi
done
mkdir -p "$DAG_HOME/tokenizer"
case "${DAG_MODEL:-}" in
    1b-dagformer|600m-dagformer|600m-baseline)
        DAG_RUN=$DAG_ROOT/runs/$DAG_MODEL-complete
        DAG_RUN_META=$DAG_HOME/run_metadata/$DAG_MODEL-complete
        mkdir -p "$DAG_RUN" "$DAG_RUN_META"
        for DAG_META_FILE in resume.yaml launch.json metrics.csv; do
            if [ ! -e "$DAG_RUN/$DAG_META_FILE" ] && [ ! -L "$DAG_RUN/$DAG_META_FILE" ]; then
                ln -s "$DAG_RUN_META/$DAG_META_FILE" "$DAG_RUN/$DAG_META_FILE"
            fi
        done
        ;;
esac
cd "$DAG_CODE"
export DAG_ROOT DAG_CODE DAG_HOME
export PYTHONPATH="$DAG_CODE"
export HF_HOME="$DAG_HOME/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PIP_CACHE_DIR="$DAG_HOME/cache/pip"
export XDG_CACHE_HOME="$DAG_HOME/cache/xdg"
export WANDB_DIR="$DAG_HOME/wandb"
mkdir -p "$WANDB_DIR"
unset TRANSFORMERS_CACHE
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
printf 'Job %s on %s, code %s\n' "$SLURM_JOB_ID" "$(hostname)" "$DAG_REVISION"
