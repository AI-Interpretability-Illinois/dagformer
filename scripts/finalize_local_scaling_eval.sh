#!/bin/bash
# Finish the submitted local 600M generation pair and archive its full artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
DAG_PYTHON=${DAG_EVAL_PYTHON:-/scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python}
DAG_OUT=experiments/results/scaling_eval_20260917
DAG_BF=/work/hdd/bfqt/yurenh2/dagformer-eval-20260917
DAG_BIRO=/work/hdd/biro/yurenh2/dagformer-20260917
DAG_BASE_PID=${1:?Pass the baseline generation PID}
DAG_DAG_PID=${2:?Pass the DAGFormer generation PID}
while kill -0 "$DAG_BASE_PID" 2>/dev/null || kill -0 "$DAG_DAG_PID" 2>/dev/null; do
    sleep 30
done
test -f "$DAG_OUT/gsm8k_full_600m/600m-baseline__custom.json"
test -f "$DAG_OUT/gsm8k_full_600m/600m-dagformer__custom.json"
"$DAG_PYTHON" scripts/summarize_paired_eval.py --dir "$DAG_OUT/gsm8k_full_600m"
"$DAG_PYTHON" scripts/summarize_generation_audit.py --dir "$DAG_OUT/gsm8k_full_600m"
rsync -a "$DAG_OUT/" "delta:$DAG_BF/$DAG_OUT/"
ssh delta sbatch --parsable --account=biro-delta-gpu --partition=gpuA40x4 \
    --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-node=1 --mem=4G --time=00:10:00 \
    --job-name=archive-600m-eval \
    --output="$DAG_BF/logs/scaling_20260917/archive_600m_%j.log" \
    --wrap="'mkdir -p $DAG_BIRO/results/interim_600m && rsync -a $DAG_BF/$DAG_OUT/ $DAG_BIRO/results/interim_600m/'"
date -u '+Local 600M generation pair finalized at %Y-%m-%dT%H:%M:%SZ'
