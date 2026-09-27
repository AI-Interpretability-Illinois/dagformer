#!/bin/bash
# Evaluate the finished 300M fourway_corrected and fourway_modular runs on the
# held-out caches and write the 1B architecture decision (lowest Dolma NLL):
#   experiments/pretrain_1b/CHOSEN                 (winning routing_mode on line 1)
#   experiments/pretrain_1b/CHOSEN_NOT_<loser>     (skipif marker for the loser's 1B item)
set -euo pipefail
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1
D=/work/hdd/bfqt/xiaocong/dagformer_pruning/data
M=/work/hdd/bfqt/xiaocong/dagformer_300m
mkdir -p experiments/pretrain_300m
python3 scripts/eval_pretrained.py \
    --model $M/300m_fourway_corrected --config configs/pretrain_300m/300m_fourway_corrected_mb4.yaml \
    --model $M/300m_fourway_modular   --config configs/pretrain_300m/300m_fourway_modular_mb4.yaml \
    --eval dolma=$D/dolma_v1_7_21b/eval_cache.pt --eval wikitext2=$D/wikitext2/eval_cache.pt \
    --eval mathinstruct=$D/mathinstruct/eval_cache.pt --eval gsm8k=$D/gsm8k/eval_cache.pt \
    --out experiments/pretrain_300m/eval_300m.json --decide experiments/pretrain_1b
