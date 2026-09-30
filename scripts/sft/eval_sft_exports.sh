#!/bin/bash
# Benchmark every finished instruction-tuning export (runs/<task>/final, shared-model layout)
# with the project's lm-eval harness. Incremental: existing result files are skipped, so it can
# be re-run as more SFT tasks finish. One GPU; ~20-40 min per 300M model per suite.
#   bash scripts/sft/eval_sft_exports.sh [name-filter regex, default: pt_dagformer_300m] [suites, default: "reasoning core"]
set -uo pipefail
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHON=${PYTHON:-/u/xiaocong/anaconda3/envs/modularity/bin/python}
FILTER=${1:-pt_dagformer_300m}; SUITES=${2:-"reasoning core"}
RUNS=/work/hdd/bfqt/xiaocong/dagformer_sft/runs; EXPORTS=/work/hdd/bfqt/xiaocong/dagformer_sft/exports
mkdir -p "$EXPORTS" experiments/results/lmeval/sft
n=0
for d in "$RUNS"/*/; do
    t=$(basename "$d"); [[ "$t" =~ $FILTER ]] || continue
    [ -f "$d/DONE" ] && [ -f "$d/final/config.yaml" ] && [ -f "$d/final/checkpoint.pt" ] || continue
    ln -sfn "$d/final" "$EXPORTS/$t"; n=$((n+1))
done
echo "$n finished exports match '$FILTER'"; [ "$n" -gt 0 ] || exit 0
for suite in $SUITES; do
    "$PYTHON" experiments/results/lmeval/run_eval.py --models-root "$EXPORTS" --model all --suite "$suite" \
        --out-dir experiments/results/lmeval/sft --gen-limit 200 --save-samples
done
