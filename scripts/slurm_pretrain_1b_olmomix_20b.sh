#!/bin/bash
#SBATCH --job-name=1b-base20b
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=1800G
#SBATCH --time=48:00:00
#SBATCH --output=logs/pretrain_1b_olmomix_20b_%j.out
#SBATCH --error=logs/pretrain_1b_olmomix_20b_%j.err
#SBATCH --signal=B:USR1@180

set -eo pipefail

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs /work/hdd/bfqt/dagformer_checkpoints/pretrain_1b_baseline_olmomix_20b

# HF cache lives on /work — /projects is at 98% (500G hard quota) and a save
# there would silently fail mid-training.
export HF_HOME=/work/hdd/bfqt/yurenh2/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH

# HTTP download tolerance — HF streams sometimes stall or return 503s. The
# in-loop retry (MAX_RETRIES=999999 in src/data/dolma.py) handles transient
# failures; this just keeps individual hub requests from timing out too fast.
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=0   # disable hf_transfer (slow on Delta)
export HF_DATASETS_TIMEOUT=300

CONFIG=configs/pretrain_1b_baseline_olmomix_20b.yaml

# Auto-resubmit chain: if walltime runs out we re-queue ourselves so resume
# picks up from the latest checkpoint on the next run.
AUTORESUBMIT=${AUTORESUBMIT:-1}
trap 'echo "Received SIGUSR1 — trainer should save & exit cleanly."; \
      if [ "$AUTORESUBMIT" = "1" ]; then \
          echo "Re-submitting self for resume..."; \
          sbatch --dependency=afterany:$SLURM_JOB_ID $0; \
      fi' SIGUSR1

echo "Config:    $CONFIG"
echo "Save dir:  /work/hdd/bfqt/dagformer_checkpoints/pretrain_1b_baseline_olmomix_20b"
echo "Started:   $(date)"
echo "Job ID:    $SLURM_JOB_ID"

torchrun \
    --nproc_per_node=8 \
    --master_port=29520 \
    scripts/pretrain_baseline.py \
    --config "$CONFIG"

echo "Run completed cleanly at $(date)."
