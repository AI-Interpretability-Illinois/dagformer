#!/bin/bash
# Common held-out re-evaluation of all finished runs on this machine (queue task on Delta;
# run by hand on timan1 / timan108 with EVAL_ROOT pointing at the copied caches).
set -euo pipefail
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1
MACHINE=${1:-delta}
D=${EVAL_ROOT:-/work/hdd/bfqt/xiaocong/dagformer_pruning/data}
"${PYTHON:-python3}" scripts/scaling_common_eval.py --machine "$MACHINE" \
    --eval dolma21b=$D/dolma_v1_7_21b/eval_cache.pt --eval wikitext2=$D/wikitext2/eval_cache.pt \
    --eval mathinstruct=$D/mathinstruct/eval_cache.pt --eval gsm8k=$D/gsm8k/eval_cache.pt "${@:2}"
