#!/bin/bash
#SBATCH --job-name=dag-complete
#SBATCH --account=biro-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=36:00:00
#SBATCH --output=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917/logs/scaling_20260917/train_%x_%j.log
#SBATCH --error=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917/logs/scaling_20260917/train_%x_%j.log
set -euo pipefail
source "${DAG_CODE_SOURCE:-/work/hdd/bfqt/yurenh2/dagformer-eval-20260917}/scripts/biro_job_common.sh"
DAG_MODEL=${DAG_MODEL:?Choose 1b-dagformer, 600m-dagformer or 600m-baseline}
DAG_CONFIG=$("$DAG_PYTHON" scripts/prepare_scaling_resume.py --root "$DAG_ROOT" --model "$DAG_MODEL")
if [[ "$DAG_MODEL" == *baseline ]]; then
    DAG_TRAINER=scripts/pretrain_baseline.py
else
    DAG_TRAINER=scripts/pretrain_dagformer.py
fi
cat "$DAG_ROOT/runs/$DAG_MODEL-complete/launch.json"
"$DAG_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "$DAG_TRAINER" --config "$DAG_CONFIG"
DAG_CHECKPOINT=$("$DAG_PYTHON" scripts/prepare_scaling_resume.py --root "$DAG_ROOT" \
    --model "$DAG_MODEL" --check-complete)
"$DAG_PYTHON" scripts/stage_scaling_checkpoint.py --checkpoint "$DAG_CHECKPOINT" \
    --config "$DAG_CONFIG" --out-dir "$DAG_ROOT/checkpoints/eval/$DAG_MODEL" \
    --tokens-per-update 524288
