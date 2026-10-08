#!/bin/bash
# Submit the post-SFT benchmark pass (scripts/sft/eval_sft_exports.sh) as a 1-GPU job in the
# reservation, exactly once, as soon as all instruction-tuning tasks matching FILTER are DONE.
# Called by every SFT filler after a task finishes and by the GPU coordinator; safe to call often.
#   bash scripts/slurm/submit_sft_eval.sh [filter regex, default pt_dagformer_300m] [needed count, default 6]
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/resv_lib.sh"
REPO_DIR=${REPO_DIR:-/u/xiaocong/dagformer}
FILTER=${1:-pt_dagformer_300m}; NEEDED=${2:-6}
RUNS=/work/hdd/bfqt/xiaocong/dagformer_sft/runs
MARK=/work/hdd/bfqt/xiaocong/dagformer_sft/eval_submitted_${FILTER}
[ -f "$MARK" ] && exit 0
done_n=$(ls -d "$RUNS"/*/ 2>/dev/null | while read -r d; do t=$(basename "$d"); [[ "$t" =~ $FILTER ]] && [ -f "$d/DONE" ] && echo x; done | wc -l)
[ "$done_n" -ge "$NEEDED" ] || { echo "sft-eval: $done_n/$NEEDED '$FILTER' tasks DONE, not yet"; exit 0; }
cd "$REPO_DIR"; mkdir -p logs
jid=$(sbatch_with_fallback --parsable --reservation="$RESV" --partition=gpuA100x4 --nodes=1 --gpus-per-node=1 \
      --ntasks-per-node=1 --cpus-per-task=16 --mem=40G --time=06:00:00 --job-name="sft-eval-${FILTER}" \
      --chdir="$REPO_DIR" --output="logs/sft_eval_${FILTER}_%j.out" \
      --wrap "bash scripts/sft/eval_sft_exports.sh '$FILTER'") || { echo "sft-eval: submit failed" >&2; exit 1; }
echo "$jid $(date '+%F %T')" > "$MARK"
log "sft-eval: all $NEEDED '$FILTER' tasks DONE -> submitted benchmark job $jid"
