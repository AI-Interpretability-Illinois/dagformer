#!/bin/bash
#SBATCH --job-name=dag-freeze-600m
#SBATCH --account=biro-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917/logs/scaling_20260917/freeze_600m_%j.log
#SBATCH --error=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917/logs/scaling_20260917/freeze_600m_%j.log
set -euo pipefail
source "${DAG_CODE_SOURCE:-/work/hdd/bfqt/yurenh2/dagformer-eval-20260917}/scripts/biro_job_common.sh"
DAG_DATA_ENV=$DAG_ROOT/envs/dolma-data
if [ ! -x "$DAG_DATA_ENV/bin/python" ]; then
    "$DAG_PYTHON" -m venv --system-site-packages "$DAG_DATA_ENV"
fi
"$DAG_DATA_ENV/bin/python" -m pip install 'datasets==3.6.0'
"$DAG_DATA_ENV/bin/python" scripts/freeze_dolma_continuation.py \
    --out-dir "$DAG_ROOT/data/pretok/dolma_600m_continuation" \
    --tokenizer "$DAG_ROOT/tokenizer"
