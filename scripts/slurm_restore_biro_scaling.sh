#!/bin/bash
#SBATCH --job-name=dag-restore
#SBATCH --account=biro-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917/logs/scaling_20260917/restore_%j.log
#SBATCH --error=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917/logs/scaling_20260917/restore_%j.log

set -euo pipefail
source "${DAG_CODE_SOURCE:-/work/hdd/bfqt/yurenh2/dagformer-eval-20260917}/scripts/biro_job_common.sh"
DAG_OLD_MODELS=/work/hdd/bfqt/dagformer_checkpoints
mkdir -p "$DAG_ROOT/data" "$DAG_ROOT/checkpoints"
id
lfs project -d /work/hdd/biro /projects/biro
git rev-parse HEAD

rsync -a "$DAG_SOURCE/checkpoints/pr_sync_20260917/tokenizer/" "$DAG_ROOT/tokenizer/"
rsync -a "$DAG_SOURCE/checkpoints/pr_sync_20260917/eval_corpora/" "$DAG_ROOT/data/eval_corpora/"

"$DAG_PYTHON" scripts/restore_olmo_mix.py \
    --out-dir "$DAG_ROOT/data/olmo-mix-1124" --workers 4 &
DAG_DOWNLOAD_PID=$!
trap 'kill "$DAG_DOWNLOAD_PID" 2>/dev/null || true' EXIT

"$DAG_PYTHON" scripts/stage_scaling_checkpoint.py \
    --checkpoint "$DAG_OLD_MODELS/fourway_1b_dagformer_olmomix_20b/checkpoint_step31000.pt" \
    --config configs/fourway_1b_dagformer_olmomix_20b.yaml \
    --out-dir "$DAG_ROOT/checkpoints/source/1b-dagformer" \
    --keep-optimizer --tokens-per-update 524288
"$DAG_PYTHON" scripts/stage_scaling_checkpoint.py \
    --checkpoint "$DAG_OLD_MODELS/pretrain_1b_baseline_olmomix_20b/checkpoint_step38160.pt" \
    --config configs/pretrain_1b_baseline_olmomix_20b.yaml \
    --out-dir "$DAG_ROOT/checkpoints/eval/1b-baseline" --tokens-per-update 524288
"$DAG_PYTHON" scripts/stage_scaling_checkpoint.py \
    --checkpoint "$DAG_OLD_MODELS/fourway_600m_chinchilla/checkpoint_step14000.pt" \
    --config configs/fourway_600m_chinchilla.yaml \
    --out-dir "$DAG_ROOT/checkpoints/source/600m-dagformer" \
    --keep-optimizer --tokens-per-update 524288 --training-data-source stream
"$DAG_PYTHON" scripts/stage_scaling_checkpoint.py \
    --checkpoint "$DAG_OLD_MODELS/pretrain_600m_baseline/checkpoint_step14000.pt" \
    --config configs/pretrain_600m_baseline.yaml \
    --out-dir "$DAG_ROOT/checkpoints/source/600m-baseline" \
    --keep-optimizer --tokens-per-update 524288 --training-data-source stream

wait "$DAG_DOWNLOAD_PID"
trap - EXIT
"$DAG_PYTHON" - <<'PY'
import importlib.metadata, json, os
from datetime import datetime, timezone
from pathlib import Path
root = Path(os.environ['DAG_HOME'])
record = {
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'slurm_job_id': os.environ['SLURM_JOB_ID'],
    'account': os.environ['SLURM_JOB_ACCOUNT'],
    'root': os.environ['DAG_ROOT'],
    'home_root': str(root),
    'source_revision': os.environ['DAG_CODE_REVISION'],
    'versions': {p: importlib.metadata.version(p) for p in
                 ['torch', 'transformers', 'datasets', 'huggingface-hub']},
}
(root / 'provenance' / 'restoration.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
PY
