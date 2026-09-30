#!/bin/bash
#SBATCH --job-name=dag-six-axis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
set -uo pipefail
cd "${SIX_CODE:?}"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${SIX_HOME:?}/cache/huggingface"
export XDG_CACHE_HOME="$SIX_HOME/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$SIX_HOME/cache/torchinductor"
export TMPDIR="$SIX_HOME/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
runner=""
request_checkpoint() {
    if [[ -n "$runner" ]]; then kill -USR1 "$runner" 2>/dev/null || true; fi
}
trap request_checkpoint USR1 TERM
"${SIX_PYTHON:?}" scripts/run_six_axis.py --manifest "${SIX_MANIFEST:?}" \
    --task "$SLURM_ARRAY_TASK_ID" --minutes 220 &
runner=$!
wait "$runner"
rc=$?
if kill -0 "$runner" 2>/dev/null; then wait "$runner"; rc=$?; fi
if [[ "$rc" -eq 75 ]]; then
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit $?
fi
exit "$rc"
