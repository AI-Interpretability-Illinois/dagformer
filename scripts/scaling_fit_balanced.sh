#!/bin/bash
# D3: refit the scaling analysis on the source-balanced Dolma held-out set WITHOUT touching the paper's WikiText-2
# outputs in experiments/scaling/. Copies the inputs into experiments/scaling/balanced/ and runs scaling_fit.py there
# with --eval-key dolma_balanced; compare balanced/results.md with ../results.md (gaps, effective-parameter and
# compute multipliers; the pre-registered threshold in experiments/README.md is 0.1 on the multipliers).
set -euo pipefail
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
R=experiments/scaling/balanced
mkdir -p "$R"
cp experiments/scaling/runs.yaml experiments/scaling/collected_*.json experiments/scaling/common_eval_*.json "$R/"
"${PYTHON:-python3}" scripts/scaling_fit.py --root "$R" --eval-key dolma_balanced "$@"
echo "balanced-key fit written under $R (results.md, scaling_fits.json, runs_table.md, figures)"
