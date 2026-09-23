#!/bin/bash
#SBATCH --job-name=locality_prep
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=00:20:00
#SBATCH --output=logs/locality_prep_%j.out
#SBATCH --error=logs/locality_prep_%j.err
# Experiment 1 prep: split 200 eval windows off the Dolma slice and stage
# eval_cache.pt into every arm's save_dir. Submit with
# --dependency=afterany:<pretok job ids>; the arm jobs depend on this (afterok).
set -eu
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
D=/work/hdd/bfqt/xiaocong/dagformer_pruning/data/dolma_v1_7_1p7b
C=/work/hdd/bfqt/xiaocong/dagformer_pruning/checkpoints/locality
[ -f "$D/index.json" ] || { echo "no index.json in $D (pretokenize failed?)"; exit 1; }
if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$D/index.json')).get('eval_split_off') else 1)"; then
    python3 scripts/split_eval_from_mmap.py --index-dir "$D" --n-windows 200 --out "$D/eval_cache.pt"
fi
for arm in global local both per_layer dense; do mkdir -p "$C/$arm"; cp "$D/eval_cache.pt" "$C/$arm/eval_cache.pt"; done
python3 -c "import json; d=json.load(open('$D/index.json')); print('train tokens', d['total_tokens'], 'eval', d['eval_split_off'])"
