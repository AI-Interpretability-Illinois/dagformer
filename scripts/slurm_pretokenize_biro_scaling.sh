#!/bin/bash
#SBATCH --job-name=dag-pretok-1b
#SBATCH --account=biro-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917/logs/scaling_20260917/pretok_1b_%j.log
#SBATCH --error=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917/logs/scaling_20260917/pretok_1b_%j.log
set -euo pipefail
source "${DAG_CODE_SOURCE:-/work/hdd/bfqt/yurenh2/dagformer-eval-20260917}/scripts/biro_job_common.sh"
"$DAG_PYTHON" scripts/pretokenize_olmo_ordered.py \
    --manifest "$DAG_ROOT/data/olmo-mix-1124/download_manifest.json" \
    --out-dir "$DAG_ROOT/data/pretok/olmo_mix_21b_reconstructed" \
    --cache-dir "$DAG_ROOT/data/tokenized_files/olmo_mix" \
    --tokenizer-id "$DAG_ROOT/tokenizer" --workers 12 \
    --reconstruction experiments/results/scaling_audit_20260917/olmo_reconstruction.json

