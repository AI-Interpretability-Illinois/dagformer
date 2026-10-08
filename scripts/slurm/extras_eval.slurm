#!/bin/bash
#SBATCH --job-name=extras_eval
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/extras_eval_%j.out
# Common held-out evaluation + collect + refit over every finished run in experiments/scaling/runs.yaml
# (used after the post-hoc 150M runs; the 300M MUDDFormer chain runs the same three commands as its tail task).
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# EXTRA passes through to scaling_common_eval.py, e.g. EXTRA="--only extras_150m_denseformer --force" for a re-eval.
bash scripts/scaling_common_eval.sh delta ${EXTRA:-} && python3 scripts/scaling_collect.py --machine delta && python3 scripts/scaling_fit.py
echo "exit rc=$? at $(date)"
