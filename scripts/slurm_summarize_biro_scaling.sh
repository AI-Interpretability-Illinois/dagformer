#!/bin/bash
#SBATCH --job-name=dag-scale-summary
#SBATCH --account=biro-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917/logs/scaling_20260917/summary_%j.log
#SBATCH --error=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917/logs/scaling_20260917/summary_%j.log
set -euo pipefail
source "${DAG_CODE_SOURCE:-/work/hdd/bfqt/yurenh2/dagformer-eval-20260917}/scripts/biro_job_common.sh"
DAG_OUT=$DAG_ROOT/results/completed_scaling
DAG_SIZE=${DAG_SIZE:?Choose 600m or 1b}
DAG_PYTHON=$DAG_ROOT/envs/evaluation/bin/python
for DAG_SUITE in standard gsm8k_full; do
    "$DAG_PYTHON" scripts/summarize_paired_eval.py --dir "$DAG_OUT/${DAG_SUITE}_$DAG_SIZE"
done
"$DAG_PYTHON" scripts/compare_eval_variants.py --reference-dir "$DAG_OUT/standard_$DAG_SIZE" \
    --variant-dir "$DAG_OUT/frozen_predictor_$DAG_SIZE"
rsync -a --exclude samples/ "$DAG_OUT/" \
    "$DAG_SOURCE/experiments/results/completed_scaling_20260917/"
