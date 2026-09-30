#!/bin/bash
#SBATCH --job-name=scaling_eval_resv
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --reservation=sup-30781
#SBATCH --nodelist=gpua047
#SBATCH --output=logs/scaling_eval_%j.out
# One-off common re-evaluation of the finished Delta runs (the reserved-node queue re-runs the same
# script after the last pretraining item; already-evaluated runs are skipped).
cd /u/xiaocong/dagformer
source ~/.bashrc; conda activate modularity 2>/dev/null || true
bash scripts/scaling_common_eval.sh delta
python3 scripts/scaling_collect.py --machine delta && python3 scripts/scaling_fit.py
