#!/bin/bash
#SBATCH --job-name=1b-4way20b
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=1800G
#SBATCH --time=48:00:00
#SBATCH --output=logs/fourway_1b_olmomix_20b_%j.out
#SBATCH --error=logs/fourway_1b_olmomix_20b_%j.err
#SBATCH --signal=B:USR1@180

set -eo pipefail

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs /work/hdd/bfqt/dagformer_checkpoints/fourway_1b_dagformer_olmomix_20b

export HF_HOME=/work/hdd/bfqt/yurenh2/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH

export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_TIMEOUT=300

CONFIG=configs/fourway_1b_dagformer_olmomix_20b.yaml

AUTORESUBMIT=${AUTORESUBMIT:-1}
trap 'echo "Received SIGUSR1 — trainer should save & exit cleanly."; \
      if [ "$AUTORESUBMIT" = "1" ]; then \
          echo "Re-submitting self for resume..."; \
          sbatch --dependency=afterany:$SLURM_JOB_ID $0; \
      fi' SIGUSR1

echo "Config:    $CONFIG"
echo "Save dir:  /work/hdd/bfqt/dagformer_checkpoints/fourway_1b_dagformer_olmomix_20b"
echo "Started:   $(date)"
echo "Job ID:    $SLURM_JOB_ID"

torchrun \
    --nproc_per_node=8 \
    --master_port=29521 \
    scripts/pretrain_dagformer.py \
    --config "$CONFIG"

echo "Run completed cleanly at $(date)."
