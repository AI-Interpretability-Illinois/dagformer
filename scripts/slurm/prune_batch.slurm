#!/bin/bash
#SBATCH --job-name=prune_batch
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=logs/prune_batch_%j.out
#SBATCH --error=logs/prune_batch_%j.err

# Run several prune-finetune runs back to back on one GPU (the per-user job
# submit limit makes one-job-per-run impractical). RUNLIST is a text file,
# one run per line:  <config.yaml> <override> <override> ...
# Runs whose save_dir already holds summary.json are skipped, so the same
# list can be resubmitted after a time-out; an interrupted run resumes from
# <save_dir>/latest.
#   RUNLIST=experiments/pruning/runlists/75m_a.txt sbatch scripts/slurm/prune_batch.slurm
set -u
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
mkdir -p logs
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
RUNLIST="${RUNLIST:?set RUNLIST=<file>}"
rc=0
while read -r cfg rest; do
    [ -z "$cfg" ] && continue
    case "$cfg" in \#*) continue;; esac
    args=(); sd=""
    for ov in $rest; do args+=(--override "$ov"); case "$ov" in save_dir=*) sd="${ov#save_dir=}";; esac; done
    if [ -n "$sd" ] && [ -f "$sd/summary.json" ]; then echo "skip (done): $sd"; continue; fi
    echo "=== $cfg $rest ($(date)) ==="
    python3 scripts/prune_finetune.py --config "$cfg" "${args[@]}" || { rc=$?; echo "FAILED rc=$rc: $cfg $rest"; }
done < "$RUNLIST"
echo "batch done rc=$rc ($(date))"
exit $rc
