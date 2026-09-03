#!/bin/bash
#SBATCH --job-name=dolma-run
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_a40:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dolma_run_%j.out
#SBATCH --error=logs/dolma_run_%j.err
#SBATCH --signal=B:USR1@180

# Generic single-node 4-GPU launcher for the Dolma-tier (75M-600M) runs on the
# ABUNDANT A40x4 partition (H200/A100x8 are queue-locked). Parameterized:
#   sbatch --export=ALL,SCRIPT=scripts/pretrain_denseformer.py,CONFIG=configs/pretrain_75m_denseformer.yaml scripts/slurm_dolma_run.sh
# All configs are 4-GPU geometry (524288 tok/step). Auto-resubmits on preemption.

set -uo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/work/hdd/bfqt/yurenh2/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export HF_XET_NATIVE=0
export HF_HUB_DISABLE_XET=1

SCRIPT=${SCRIPT:-scripts/pretrain_baseline.py}
CONFIG=${CONFIG:?set CONFIG=configs/....yaml}
PORT=$((29000 + RANDOM % 2000))

AUTORESUBMIT=${AUTORESUBMIT:-1}
trap 'echo "SIGUSR1 — trainer saves & exits"; \
      if [ "$AUTORESUBMIT" = "1" ]; then \
          sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL,SCRIPT=$SCRIPT,CONFIG=$CONFIG $0; \
      fi' SIGUSR1

echo "Script:  $SCRIPT"
echo "Config:  $CONFIG"
echo "Job:     $SLURM_JOB_ID   Node: $SLURMD_NODENAME   $(date)"
torchrun --nproc_per_node=4 --master_port="$PORT" "$SCRIPT" --config "$CONFIG"
rc=$?
if [ "$rc" -eq 0 ]; then
  echo "Run completed cleanly at $(date)."
else
  echo "Run FAILED (torchrun exit $rc) at $(date)."
fi
exit "$rc"
